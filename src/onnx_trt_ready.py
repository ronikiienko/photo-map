"""Make the fused vision encoder readable by TensorRT: only standard ONNX operators.

TensorRT cannot read two of ONNX Runtime's own operators that the fused encoder uses, and would hand those nodes back to the CUDA
provider, cutting the graph into ~130 pieces. This replaces them with plain operators, which TensorRT fuses and tunes by itself:

  SimplifiedLayerNormalization (113)  ->  x * 1/sqrt(mean(x^2) + eps) * scale, the mean in float32
  MultiHeadAttention (16)             ->  split heads, Q.K^T (+ the mask bias), softmax, .V, merge heads

    python src/onnx_trt_ready.py     # writes vision_encoder_fp16_trt.onnx next to the fused one (weights are shared)
"""

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

import config

DIR = config.ROOT / ".cache" / "onnx" / "embeddinggemma-2" / "onnx"


def ints(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def main():
    m = onnx.load(str(DIR / "vision_encoder_fp16_fused.onnx"), load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    new_init, nodes = [], []
    eps_name = "trt_eps"
    new_init += [numpy_helper.from_array(np.array(1e-6, dtype=np.float32), eps_name), ints("trt_last", [-1])]
    counts = {"norm": 0, "attention": 0}

    for n in g.node:
        if n.op_type == "SimplifiedLayerNormalization":
            k = counts["norm"] = counts["norm"] + 1
            eps = next(a.f for a in n.attribute if a.name == "epsilon")
            assert next(a.i for a in n.attribute if a.name == "axis") == -1
            p = f"trt_norm{k}_"
            x, scale = n.input[0], n.input[1]
            new_init.append(numpy_helper.from_array(np.array(eps, dtype=np.float32), p + "eps"))
            nodes += [
                helper.make_node("Cast", [x], [p + "xf"], to=T.FLOAT, name=p + "cast_in"),
                helper.make_node("Mul", [p + "xf", p + "xf"], [p + "sq"], name=p + "sq"),
                helper.make_node("ReduceMean", [p + "sq", "trt_last"], [p + "mean"], keepdims=1, name=p + "mean"),
                helper.make_node("Add", [p + "mean", p + "eps"], [p + "var"], name=p + "add"),
                helper.make_node("Sqrt", [p + "var"], [p + "std"], name=p + "sqrt"),
                helper.make_node("Div", [p + "xf", p + "std"], [p + "normed"], name=p + "div"),
                helper.make_node("Cast", [p + "normed"], [p + "half"], to=T.FLOAT16, name=p + "cast_out"),
                helper.make_node("Mul", [p + "half", scale], [n.output[0]], name=p + "scale"),
            ]
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

    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_init)
    # nodes that only fed the replaced attention's Expand are now unused; drop what nothing reads
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    while True:
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead:
            break
        for n in dead:
            g.node.remove(n)
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    del g.value_info[:]
    out = str(DIR / "vision_encoder_fp16_trt.onnx")
    onnx.save(m, out)
    onnx.checker.check_model(out)  # by path: it looks for the weights next to the file
    print(f"{counts['norm']} norms and {counts['attention']} attentions rewritten, {len(g.node)} nodes in all")


if __name__ == "__main__":
    main()
