"""Make the exported vision encoder use ONNX Runtime's fused attention.

The export folds the padding mask into the dot product by widening every attention head from 64 to 68 (a constant 1 in the
query, the key's mask bias in the key, 4 zero columns in the value, sliced off afterwards). A head width that isn't a multiple of 8
makes ONNX Runtime fall back to the unfused path (the whole 1260 x 1260 score matrix, a separate softmax). This puts the heads back
to 64 and passes the mask as MultiHeadAttention's attention_bias instead, which its memory-efficient kernel takes -
if the sequence length is a multiple of 8 (pad pixel_values with zeros and position ids with -1 up to 1264).

    python src/onnx_fuse_attention.py        # writes vision_encoder_fp16_fused.onnx next to the original (weights are shared)
"""

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

import config

DIR = config.ROOT / ".cache" / "onnx" / "embeddinggemma-2" / "onnx"
HEAD, WIDE = 64, 68


def main():
    m = onnx.load(str(DIR / "vision_encoder_fp16.onnx"), load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    small = {i.name: numpy_helper.to_array(i) for i in g.initializer if i.data_location != onnx.TensorProto.EXTERNAL}
    for n in g.node:
        if n.op_type == "Constant" and n.attribute[0].t.data_location != onnx.TensorProto.EXTERNAL:
            small[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    # the mask as a key bias [batch, seq] (0, or -32752 for padding), made once
    key_bias = "_to_copy"
    assert key_bias in prod and prod[key_bias].op_type == "Cast"
    new_nodes = [
        helper.make_node("Unsqueeze", [key_bias, "fuse_axes12"], ["fuse_bias_b11s"]),
        helper.make_node("Shape", [key_bias], ["fuse_shape_bs"]),
        helper.make_node("Slice", ["fuse_shape_bs", "fuse_i0", "fuse_i1"], ["fuse_b"]),
        helper.make_node("Slice", ["fuse_shape_bs", "fuse_i1", "fuse_i2"], ["fuse_s"]),
        helper.make_node("Concat", ["fuse_b", "fuse_one", "fuse_s", "fuse_s"], ["fuse_target"], axis=0),
        helper.make_node("Expand", ["fuse_bias_b11s", "fuse_target"], ["fuse_attention_bias"]),
    ]
    consts = {"fuse_axes12": [1, 2], "fuse_i0": [0], "fuse_i1": [1], "fuse_i2": [2], "fuse_one": [1]}
    g.initializer.extend(numpy_helper.from_array(np.array(v, np.int64), k) for k, v in consts.items())

    dead = set()  # nodes to drop
    swap = {}  # tensor -> the tensor that takes its place
    mhas = [n for n in g.node if n.op_type == "MultiHeadAttention"]
    assert len(mhas) == 16
    for mha in mhas:
        q, k, v = mha.input[:3]
        for name in (q, v):  # Reshape <- Pad(x, pads, 1.0): back to x
            reshape = prod[name]
            pad = prod[reshape.input[0]]
            assert reshape.op_type == "Reshape" and pad.op_type == "Pad"
            swap[reshape.input[0]] = pad.input[0]
            dead.add(pad.name)
        reshape = prod[k]
        cat = prod[reshape.input[0]]
        assert reshape.op_type == "Reshape" and cat.op_type == "Concat"
        swap[cat.output[0]] = cat.input[0]
        dead.add(cat.name)
        del mha.input[:]
        mha.input.extend([q, k, v, "", "", "fuse_attention_bias"])
    for n in g.node:
        for i, name in enumerate(n.input):
            if name in swap:
                n.input[i] = swap[name]

    # every head is 64 wide again: the reshape that merges heads (12 x 68 = 816 -> 768) and the one that splits them (68 -> 64)
    def set_init(name, value):
        for init in g.initializer:
            if init.name == name:
                init.CopyFrom(numpy_helper.from_array(np.array(value, small[name].dtype), name))
                return
        raise KeyError(name)

    assert small["val_270"].tolist() == [12 * WIDE] and small["val_286"].tolist() == [WIDE]
    set_init("val_270", [12 * HEAD])
    set_init("val_286", [HEAD])

    kept = [n for n in g.node if n.name not in dead]
    at = next(i for i, n in enumerate(kept) if key_bias in n.output) + 1
    kept[at:at] = new_nodes
    del g.node[:]
    g.node.extend(kept)
    out = DIR / "vision_encoder_fp16_fused.onnx"
    onnx.save(m, str(out))
    print("wrote", out)


if __name__ == "__main__":
    main()
