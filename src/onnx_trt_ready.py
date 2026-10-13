"""Make the fused vision encoder readable by TensorRT: only standard ONNX operators.

TensorRT cannot read two of ONNX Runtime's own operators that the fused encoder uses, and would hand those nodes back to the CUDA
provider, cutting the graph into ~130 pieces. This replaces them with plain operators, which TensorRT fuses and tunes by itself:

  SimplifiedLayerNormalization (113)  ->  RMSNormalization (ONNX opset 23)
  MultiHeadAttention (16)             ->  split heads, Q.K^T (+ the mask bias), softmax, .V, merge heads

    python src/onnx_trt_ready.py     # writes vision_encoder_fp16_trt.onnx next to the fused one (weights are shared)
"""

import os

import numpy as np
import onnx
import onnx.utils
from onnx import TensorProto as T, helper, numpy_helper

import config

DIR = config.ROOT / ".cache" / "onnx" / "embeddinggemma-2" / "onnx"
SHRINK = 16.0


def ints(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def prune(g):
    """Drop the nodes nothing reads."""
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    while True:
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead:
            return
        for n in dead:
            g.node.remove(n)
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}


def main():
    m = onnx.load(str(DIR / "vision_encoder_fp16_fused.onnx"), load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    new_init, nodes = [], []
    new_init.append(numpy_helper.from_array(np.array(1 / SHRINK, dtype=np.float16), "trt_shrink"))
    counts = {"norm": 0, "attention": 0}

    for n in g.node:
        if n.op_type == "SimplifiedLayerNormalization":
            k = counts["norm"] = counts["norm"] + 1
            eps = next(a.f for a in n.attribute if a.name == "epsilon")
            assert next(a.i for a in n.attribute if a.name == "axis") == -1
            # One RMSNormalization (ONNX opset 23). Its input is first divided by SHRINK: the sum of squares inside TensorRT's float16
            # engine overflows once a value is above 256, and the row comes out as zeros. The result does not change: dividing x by s
            # and epsilon by s*s gives x / sqrt(mean(x^2) + epsilon) exactly.
            small = f"trt_norm{k}_small"
            nodes.append(helper.make_node("Mul", [n.input[0], "trt_shrink"], [small], name=f"trt_norm{k}_shrink"))
            nodes.append(helper.make_node("RMSNormalization", [small, n.input[1]], [n.output[0]], axis=-1, epsilon=eps / SHRINK**2, stash_type=1, name=f"trt_norm{k}"))
        elif n.op_type == "MultiHeadAttention":
            k = counts["attention"] = counts["attention"] + 1
            heads = next(a.i for a in n.attribute if a.name == "num_heads")
            scale = next(a.f for a in n.attribute if a.name == "scale")
            assert scale == 1, "the queries are scaled before the attention in this export"
            q, key, v, _, _, bias = n.input
            # the mask bias is an Expand of a [batch, 1, 1, length] tensor: the small one broadcasts, no need for the big one
            if bias in prod and prod[bias].op_type == "Expand":
                bias = prod[bias].input[0]
            p = f"trt_attn{k}_"
            width = 768 // heads
            new_init += [ints(p + "split", [0, 0, heads, width]), ints(p + "merge", [0, 0, heads * width])]
            nodes += [
                helper.make_node("Reshape", [q, p + "split"], [p + "q4"], name=p + "rq"),
                helper.make_node("Reshape", [key, p + "split"], [p + "k4"], name=p + "rk"),
                helper.make_node("Reshape", [v, p + "split"], [p + "v4"], name=p + "rv"),
                helper.make_node("Transpose", [p + "q4"], [p + "q"], perm=[0, 2, 1, 3], name=p + "tq"),
                helper.make_node("Transpose", [p + "k4"], [p + "kt"], perm=[0, 2, 3, 1], name=p + "tk"),
                helper.make_node("Transpose", [p + "v4"], [p + "v"], perm=[0, 2, 1, 3], name=p + "tv"),
                helper.make_node("MatMul", [p + "q", p + "kt"], [p + "scores"], name=p + "qk"),
                helper.make_node("Add", [p + "scores", bias], [p + "masked"], name=p + "mask"),
                helper.make_node("Softmax", [p + "masked"], [p + "probs"], axis=-1, name=p + "softmax"),
                helper.make_node("MatMul", [p + "probs", p + "v"], [p + "ctx"], name=p + "pv"),
                helper.make_node("Transpose", [p + "ctx"], [p + "ctxt"], perm=[0, 2, 1, 3], name=p + "tc"),
                helper.make_node("Reshape", [p + "ctxt", p + "merge"], [n.output[0]], name=p + "rc"),
            ]
        else:
            nodes.append(n)

    for o in m.opset_import:
        if o.domain == "":
            o.version = 23  # RMSNormalization is new in opset 23
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_init)
    prune(g)  # nodes that only fed the replaced attention's Expand are now unused
    del g.value_info[:]

    # TensorRT gets the transformer layers; the pooling that follows them (integer position arithmetic: floor-divide, ReduceMax, Range)
    # comes out wrong in TensorRT for some rows, so it stays on the CUDA provider as a second small model.
    hidden = "add_7097"
    whole = str(DIR / "_vision_whole.onnx")
    onnx.save(m, whole)
    pool = str(DIR / "vision_pool.onnx")
    onnx.utils.extract_model(whole, pool, [hidden, "pixel_position_ids"], ["image_features"], check_model=False)
    os.remove(whole)
    # the two parts hand over the hidden states as float32, so the program never has to handle float16 tensors
    p = onnx.load(pool, load_external_data=False)
    for i in p.graph.input:
        if i.name == hidden:
            i.name = "hidden"
            i.type.tensor_type.elem_type = T.FLOAT
    p.graph.node.insert(0, helper.make_node("Cast", ["hidden"], [hidden], to=T.FLOAT16, name="trt_pool_cast_in"))
    onnx.save(p, pool)
    del g.output[:]
    g.node.append(helper.make_node("Cast", [hidden], ["hidden"], to=T.FLOAT, name="trt_layers_cast_out"))
    g.output.append(helper.make_tensor_value_info("hidden", T.FLOAT, ["batch", "length", 768]))
    prune(g)
    out = str(DIR / "vision_encoder_fp16_trt.onnx")
    onnx.save(m, out)
    onnx.checker.check_model(out)  # by path: it looks for the weights next to the file
    print(f"{counts['norm']} norms and {counts['attention']} attentions rewritten; transformer part {len(g.node)} nodes (TensorRT), pooling in vision_pool.onnx (CUDA)")


if __name__ == "__main__":
    main()
