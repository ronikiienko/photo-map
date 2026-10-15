//! Patches: copies of a model's networks changed for one runtime, made on this computer from the files its publisher has (so no
//! changed model file is ever hosted anywhere). Each is made the first time it is needed, when the model is loaded, and kept next
//! to the downloaded files; removing the model removes them too. The weights are not copied: a patched network points at the same
//! `.onnx_data` file as the one it was made from.
//!
//! - Gemma on CUDA (`GEMMA_CUDA`): the photo encoder's attention heads back to 64 wide with the padding mask as an attention bias,
//!   so ONNX Runtime uses its fused attention.
//! - Gemma on TensorRT (`GEMMA_TENSORRT`, `GEMMA_POOL`): the same, then ONNX Runtime's own operators (113 norms, 16 attentions)
//!   written with standard ones, which TensorRT can take; the pooling after the layers is a second small network that runs on CUDA.
//! - SigLIP on TensorRT (`SIGLIP_TENSORRT`): the publisher's file (opset 14) spells every layer norm out (mean, subtract, square,
//!   ...); TensorRT squares in float16, which overflows, and the vectors come out wrong. Each becomes one LayerNormalization, its
//!   input divided by 16 first (and epsilon by 256: the same result), as for Gemma's norms.
//! - On the processor (`SIGLIP_PROCESSOR`, `GEMMA_PROCESSOR`, `GEMMA_TEXT_PROCESSOR`): the networks that run for every photo in
//!   float32 (`to_float32`). ONNX Runtime's float16 arithmetic on a processor is slower: SigLIP 256 made 6.7 photos/s in float16
//!   and 8.3 in float32 on a Ryzen 5 5600H (the same vectors); Gemma about twice as fast on an M1. Search words alone stay float16.
//! - SigLIP on CoreML (`SIGLIP_COREML`): the first layer's convolution gets its padding written out (zeros, which is what no padding
//!   means): ONNX Runtime's CoreML provider refuses a convolution without it. And every matrix product by a weight becomes a `Gemm`
//!   (`weights_as_gemm`): the provider writes a product's weight into CoreML's program as text, which CoreML parses at every load.

use onnx_protobuf::{AttributeProto, Message, ModelProto, NodeProto, TensorProto, attribute_proto::AttributeType, tensor_proto::DataType};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

pub const GEMMA_CUDA: &str = "onnx/vision_encoder_fp16.cuda.onnx";
pub const GEMMA_TENSORRT: &str = "onnx/vision_encoder_fp16.tensorrt.onnx";
pub const GEMMA_COREML: &str = "onnx/vision_encoder_fp16.coreml.onnx";
pub const GEMMA_POOL: &str = "onnx/vision_encoder_fp16.pool.onnx";
pub const SIGLIP_COREML: &str = "onnx/vision_model_fp16.coreml.onnx";
pub const SIGLIP_TENSORRT: &str = "onnx/vision_model_fp16.tensorrt.onnx";
pub const SIGLIP_PROCESSOR: &str = "onnx/vision_model_fp16.processor.onnx";
pub const GEMMA_PROCESSOR: &str = "onnx/vision_encoder_fp16.processor.onnx";
pub const GEMMA_TEXT_PROCESSOR: &str = "onnx/model_fp16.processor.onnx";

const GEMMA_VISION: &str = "onnx/vision_encoder_fp16.onnx";

/// The patched network for `name` in a model's folder, made now if it is not there yet.
pub fn ensure(dir: &Path, name: &str) -> Result<PathBuf> {
    let path = dir.join(name);
    if path.exists() {
        return Ok(path);
    }
    match name {
        GEMMA_CUDA => save(&gemma_fused(load(&dir.join(GEMMA_VISION))?)?, &path)?,
        GEMMA_TENSORRT | GEMMA_POOL | GEMMA_COREML => {
            let (layers, pool) = gemma_tensorrt(gemma_fused(load(&dir.join(GEMMA_VISION))?)?, name == GEMMA_COREML)?;
            save(&pool, &dir.join(GEMMA_POOL))?;
            save(&layers, &dir.join(if name == GEMMA_COREML { GEMMA_COREML } else { GEMMA_TENSORRT }))?;
        }
        SIGLIP_COREML => save(&siglip_coreml(load(&dir.join("onnx/vision_model_fp16.onnx"))?)?, &path)?,
        SIGLIP_TENSORRT => save(&layer_norms(load(&dir.join("onnx/vision_model_fp16.onnx"))?)?, &path)?,
        SIGLIP_PROCESSOR => save(&to_float32(&dir.join("onnx"), load(&dir.join("onnx/vision_model_fp16.onnx"))?)?, &path)?,
        GEMMA_PROCESSOR => save(&to_float32(&dir.join("onnx"), load(&dir.join(GEMMA_VISION))?)?, &path)?,
        GEMMA_TEXT_PROCESSOR => save(&to_float32(&dir.join("onnx"), load(&dir.join("onnx/model_fp16.onnx"))?)?, &path)?,
        _ => return Err(format!("no patch is called {name}")),
    }
    Ok(path)
}

/// The patched network, or if it can't be made the publisher's `fallback`, which works too (only slower).
pub fn ensure_or(dir: &Path, name: &str, fallback: &str) -> PathBuf {
    ensure(dir, name).unwrap_or_else(|e| {
        eprintln!("{name} could not be made, {fallback} is used: {e}");
        dir.join(fallback)
    })
}

fn load(path: &Path) -> Result<ModelProto> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    ModelProto::parse_from_bytes(&bytes).map_err(|e| format!("reading {}: {e}", path.display()))
}

/// Written beside and then moved into place, so a half-written file is never taken for a whole one.
fn save(model: &ModelProto, path: &Path) -> Result<()> {
    let bytes = model.write_to_bytes().map_err(|e| e.to_string())?;
    let part = path.with_extension("onnx.part");
    std::fs::write(&part, bytes).map_err(|e| format!("writing {}: {e}", part.display()))?;
    std::fs::rename(&part, path).map_err(|e| format!("writing {}: {e}", path.display()))
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], name: &str, attributes: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.into(),
        attribute: attributes,
        ..Default::default()
    }
}

fn int(name: &str, i: i64) -> AttributeProto {
    AttributeProto { name: name.into(), type_: AttributeType::INT.into(), i, ..Default::default() }
}

fn ints(name: &str, ints: &[i64]) -> AttributeProto {
    AttributeProto { name: name.into(), type_: AttributeType::INTS.into(), ints: ints.to_vec(), ..Default::default() }
}

fn float(name: &str, f: f32) -> AttributeProto {
    AttributeProto { name: name.into(), type_: AttributeType::FLOAT.into(), f, ..Default::default() }
}

fn attribute<'a>(n: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}

fn int64s(name: &str, values: &[i64]) -> TensorProto {
    TensorProto { name: name.into(), data_type: DataType::INT64 as i32, dims: vec![values.len() as i64], raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(), ..Default::default() }
}

/// The values of a small int64 tensor kept in the file itself.
fn int64_values(t: &TensorProto) -> Vec<i64> {
    if !t.int64_data.is_empty() { t.int64_data.clone() } else { t.raw_data.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect() }
}

/// Which node makes each tensor.
fn producers(nodes: &[NodeProto]) -> HashMap<String, usize> {
    nodes.iter().enumerate().flat_map(|(i, n)| n.output.iter().map(move |o| (o.clone(), i))).collect()
}

/// Drop the nodes nothing reads, and the weights no node reads.
fn prune(model: &mut ModelProto) {
    let g = model.graph.as_mut().unwrap();
    loop {
        let used: HashSet<&String> = g.node.iter().flat_map(|n| n.input.iter()).chain(g.output.iter().map(|o| &o.name)).collect();
        let keep: Vec<bool> = g.node.iter().map(|n| n.output.iter().any(|o| used.contains(o))).collect();
        if keep.iter().all(|k| *k) {
            break;
        }
        let mut it = keep.into_iter();
        g.node.retain(|_| it.next().unwrap());
    }
    let used: HashSet<String> = g.node.iter().flat_map(|n| n.input.iter().cloned()).chain(g.output.iter().map(|o| o.name.clone())).collect();
    g.initializer.retain(|t| used.contains(&t.name));
}

/// See docs/models.md. The export widens every attention head from 64 to 68 to fold the padding mask
/// into the dot product; ONNX Runtime then can't use its fused attention. This puts the heads back to 64 and gives the mask to
/// MultiHeadAttention as its attention bias.
fn gemma_fused(mut model: ModelProto) -> Result<ModelProto> {
    const KEY_BIAS: &str = "_to_copy";
    let g = model.graph.as_mut().ok_or("no graph")?;
    let prod = producers(&g.node);
    let bias_at = *prod.get(KEY_BIAS).ok_or("this is not the photo encoder the patch is for (no _to_copy)")?;
    if g.node[bias_at].op_type != "Cast" {
        return Err("this is not the photo encoder the patch is for (_to_copy)".into());
    }
    let new_nodes = vec![
        node("Unsqueeze", &[KEY_BIAS, "fuse_axes12"], &["fuse_bias_b11s"], "fuse_unsqueeze", vec![]),
        node("Shape", &[KEY_BIAS], &["fuse_shape_bs"], "fuse_shape", vec![]),
        node("Slice", &["fuse_shape_bs", "fuse_i0", "fuse_i1"], &["fuse_b"], "fuse_slice_b", vec![]),
        node("Slice", &["fuse_shape_bs", "fuse_i1", "fuse_i2"], &["fuse_s"], "fuse_slice_s", vec![]),
        node("Concat", &["fuse_b", "fuse_one", "fuse_s", "fuse_s"], &["fuse_target"], "fuse_concat", vec![int("axis", 0)]),
        node("Expand", &["fuse_bias_b11s", "fuse_target"], &["fuse_attention_bias"], "fuse_expand", vec![]),
    ];
    for (name, v) in [("fuse_axes12", &[1i64, 2][..]), ("fuse_i0", &[0]), ("fuse_i1", &[1]), ("fuse_i2", &[2]), ("fuse_one", &[1])] {
        g.initializer.push(int64s(name, v));
    }
    let mut dead = HashSet::new();
    let mut swap: HashMap<String, String> = HashMap::new();
    let attentions: Vec<usize> = (0..g.node.len()).filter(|&i| g.node[i].op_type == "MultiHeadAttention").collect();
    if attentions.len() != 16 {
        return Err(format!("expected 16 attentions, found {}", attentions.len()));
    }
    for &a in &attentions {
        let (q, k, v) = (g.node[a].input[0].clone(), g.node[a].input[1].clone(), g.node[a].input[2].clone());
        for name in [&q, &v] {
            // Reshape <- Pad(x, pads, 1.0): back to x
            let reshape = &g.node[prod[name]];
            let pad_at = prod[&reshape.input[0]];
            if reshape.op_type != "Reshape" || g.node[pad_at].op_type != "Pad" {
                return Err("the attention's query or value is not the padded one the patch is for".into());
            }
            swap.insert(reshape.input[0].clone(), g.node[pad_at].input[0].clone());
            dead.insert(pad_at);
        }
        let reshape = &g.node[prod[&k]];
        let cat_at = prod[&reshape.input[0]];
        if reshape.op_type != "Reshape" || g.node[cat_at].op_type != "Concat" {
            return Err("the attention's key is not the widened one the patch is for".into());
        }
        swap.insert(g.node[cat_at].output[0].clone(), g.node[cat_at].input[0].clone());
        dead.insert(cat_at);
        g.node[a].input = vec![q, k, v, String::new(), String::new(), "fuse_attention_bias".into()];
    }
    for n in &mut g.node {
        for i in &mut n.input {
            if let Some(to) = swap.get(i) {
                *i = to.clone();
            }
        }
    }
    // every head is 64 wide again: the reshape that merges heads (12 x 68 = 816 -> 768) and the one that splits them (68 -> 64)
    for (name, was, now) in [("val_270", 12 * 68, 12 * 64), ("val_286", 68, 64)] {
        let t = g.initializer.iter_mut().find(|t| t.name == name).ok_or(format!("no {name}"))?;
        if int64_values(t) != [was] {
            return Err(format!("{name} is not {was}"));
        }
        let fixed = int64s(name, &[now]);
        t.int64_data.clear();
        t.raw_data = fixed.raw_data;
    }
    let mut kept: Vec<NodeProto> = std::mem::take(&mut g.node).into_iter().enumerate().filter(|(i, _)| !dead.contains(i)).map(|(_, n)| n).collect();
    let at = kept.iter().position(|n| n.output.iter().any(|o| o == KEY_BIAS)).unwrap() + 1;
    kept.splice(at..at, new_nodes);
    g.node = kept;
    Ok(model)
}

/// See docs/models.md. Returns the transformer layers (for TensorRT, their result as float32 "hidden")
/// and the pooling after them (for CUDA, from "hidden" and the patches' positions).
fn gemma_tensorrt(mut model: ModelProto, spelled_norms: bool) -> Result<(ModelProto, ModelProto)> {
    // the sum of squares in a norm overflows TensorRT's float16 once a value is above 256: the input is divided by this first, and
    // epsilon by its square, which gives exactly the same result
    const SHRINK: f32 = 16.0;
    const HIDDEN: &str = "add_7097";
    let g = model.graph.as_mut().ok_or("no graph")?;
    // the mask bias is an Expand of a [batch, 1, 1, length] tensor: the small one broadcasts, no need for the big one
    let expanded: HashMap<String, String> = g.node.iter().filter(|n| n.op_type == "Expand").map(|n| (n.output[0].clone(), n.input[0].clone())).collect();
    let mut nodes = Vec::with_capacity(g.node.len() * 2);
    let mut extra = vec![int64s("trt_last_axis", &[-1]), TensorProto { name: "trt_shrink".into(), data_type: DataType::FLOAT16 as i32, raw_data: half::f16::from_f32(1.0 / SHRINK).to_le_bytes().to_vec(), ..Default::default() }];
    let (mut norms, mut attentions) = (0, 0);
    for n in std::mem::take(&mut g.node) {
        match n.op_type.as_str() {
            "SimplifiedLayerNormalization" => {
                norms += 1;
                let eps = attribute(&n, "epsilon").map_or(1e-5, |a| a.f);
                if attribute(&n, "axis").is_some_and(|a| a.i != -1) {
                    return Err("a norm that is not over the last axis".into());
                }
                let small = format!("trt_norm{norms}_small");
                nodes.push(node("Mul", &[&n.input[0], "trt_shrink"], &[&small], &format!("trt_norm{norms}_shrink"), vec![]));
                let eps = eps / (SHRINK * SHRINK);
                if spelled_norms {
                    // CoreML has no RMSNormalization: x / sqrt(mean(x^2) + epsilon) * scale, spelled out
                    let p = format!("trt_norm{norms}_");
                    let s = |x: &str| format!("{p}{x}");
                    extra.push(TensorProto { name: s("eps"), data_type: DataType::FLOAT16 as i32, raw_data: half::f16::from_f32(eps).to_le_bytes().to_vec(), ..Default::default() });
                    nodes.extend([
                        node("Mul", &[&small, &small], &[&s("square")], &s("sq"), vec![]),
                        node("ReduceMean", &[&s("square"), "trt_last_axis"], &[&s("mean")], &s("mean_of"), vec![int("keepdims", 1)]),
                        node("Add", &[&s("mean"), &s("eps")], &[&s("variance")], &s("plus_eps"), vec![]),
                        node("Sqrt", &[&s("variance")], &[&s("root")], &s("root_of"), vec![]),
                        node("Div", &[&small, &s("root")], &[&s("unit")], &s("divide"), vec![]),
                        node("Mul", &[&s("unit"), &n.input[1]], &[&n.output[0]], &s("scale"), vec![]),
                    ]);
                } else {
                    nodes.push(node("RMSNormalization", &[&small, &n.input[1]], &[&n.output[0]], &format!("trt_norm{norms}"), vec![int("axis", -1), float("epsilon", eps), int("stash_type", 1)]));
                }
            }
            "MultiHeadAttention" => {
                attentions += 1;
                let heads = attribute(&n, "num_heads").ok_or("an attention without num_heads")?.i;
                if attribute(&n, "scale").is_some_and(|a| a.f != 1.0) {
                    return Err("the queries are expected to be scaled before the attention".into());
                }
                let (q, key, v) = (&n.input[0], &n.input[1], &n.input[2]);
                let bias = expanded.get(&n.input[5]).cloned().unwrap_or_else(|| n.input[5].clone());
                let p = format!("trt_attn{attentions}_");
                let width = 768 / heads;
                extra.push(int64s(&format!("{p}split"), &[0, 0, heads, width]));
                extra.push(int64s(&format!("{p}merge"), &[0, 0, heads * width]));
                let s = |x: &str| format!("{p}{x}");
                nodes.extend([
                    node("Reshape", &[q, &s("split")], &[&s("q4")], &s("rq"), vec![]),
                    node("Reshape", &[key, &s("split")], &[&s("k4")], &s("rk"), vec![]),
                    node("Reshape", &[v, &s("split")], &[&s("v4")], &s("rv"), vec![]),
                    node("Transpose", &[&s("q4")], &[&s("q")], &s("tq"), vec![ints("perm", &[0, 2, 1, 3])]),
                    node("Transpose", &[&s("k4")], &[&s("kt")], &s("tk"), vec![ints("perm", &[0, 2, 3, 1])]),
                    node("Transpose", &[&s("v4")], &[&s("v")], &s("tv"), vec![ints("perm", &[0, 2, 1, 3])]),
                    node("MatMul", &[&s("q"), &s("kt")], &[&s("scores")], &s("qk"), vec![]),
                    node("Add", &[&s("scores"), &bias], &[&s("masked")], &s("mask"), vec![]),
                    node("Softmax", &[&s("masked")], &[&s("probs")], &s("softmax"), vec![int("axis", -1)]),
                    node("MatMul", &[&s("probs"), &s("v")], &[&s("ctx")], &s("pv"), vec![]),
                    node("Transpose", &[&s("ctx")], &[&s("ctxt")], &s("tc"), vec![ints("perm", &[0, 2, 1, 3])]),
                    node("Reshape", &[&s("ctxt"), &s("merge")], &[&n.output[0]], &s("rc"), vec![]),
                ]);
            }
            _ => nodes.push(n),
        }
    }
    if norms != 113 || attentions != 16 {
        return Err(format!("expected 113 norms and 16 attentions, found {norms} and {attentions}"));
    }
    g.node = nodes;
    g.initializer.extend(extra);
    g.value_info.clear();
    for o in &mut model.opset_import {
        if o.domain.is_empty() && !spelled_norms {
            o.version = 23; // RMSNormalization is new in opset 23
        }
    }
    prune(&mut model);

    // the pooling: what makes the output from the layers' result and the patches' positions
    let mut pool = model.clone();
    {
        let g = pool.graph.as_mut().unwrap();
        let prod = producers(&g.node);
        let mut need: HashSet<usize> = HashSet::new();
        let mut todo: Vec<String> = g.output.iter().map(|o| o.name.clone()).collect();
        while let Some(t) = todo.pop() {
            if t == HIDDEN {
                continue;
            }
            if let Some(&i) = prod.get(&t) {
                if need.insert(i) {
                    todo.extend(g.node[i].input.iter().filter(|s| !s.is_empty()).cloned());
                }
            }
        }
        let mut it = (0..g.node.len()).map(|i| need.contains(&i)).collect::<Vec<_>>().into_iter();
        g.node.retain(|_| it.next().unwrap());
        let reads: HashSet<String> = g.node.iter().flat_map(|n| n.input.iter().cloned()).collect();
        if !reads.contains(HIDDEN) {
            return Err(format!("the pooling does not start from {HIDDEN}"));
        }
        g.input.retain(|i| reads.contains(&i.name));
        let mut hidden = g.input.iter().find(|i| i.name == "pixel_position_ids").ok_or("no pixel_position_ids")?.clone();
        hidden.name = "hidden".into();
        set_float(&mut hidden, &["batch", "length", "768"]);
        g.input.insert(0, hidden);
        // the two parts hand over the hidden states as float32, so the program never handles float16 tensors
        g.node.insert(0, node("Cast", &["hidden"], &[HIDDEN], "trt_pool_cast_in", vec![int("to", DataType::FLOAT16 as i64)]));
    }
    prune(&mut pool);
    {
        let g = model.graph.as_mut().unwrap();
        let mut out = g.output[0].clone();
        out.name = "hidden".into();
        set_float(&mut out, &["batch", "length", "768"]);
        g.output = vec![out];
        g.node.push(node("Cast", &[HIDDEN], &["hidden"], "trt_layers_cast_out", vec![int("to", DataType::FLOAT as i64)]));
    }
    prune(&mut model);
    Ok((model, pool))
}

/// A graph input or output as a float32 tensor of these dimensions (numbers or names).
fn set_float(v: &mut onnx_protobuf::ValueInfoProto, dims: &[&str]) {
    use onnx_protobuf::{TensorShapeProto, tensor_shape_proto::{Dimension, dimension::Value}, type_proto::{Tensor, Value as TypeValue}};
    let shape = TensorShapeProto {
        dim: dims
            .iter()
            .map(|d| Dimension { value: Some(match d.parse::<i64>() { Ok(n) => Value::DimValue(n), Err(_) => Value::DimParam(d.to_string()) }), ..Default::default() })
            .collect(),
        ..Default::default()
    };
    let tensor = Tensor { elem_type: DataType::FLOAT as i32, shape: Some(shape).into(), ..Default::default() };
    v.type_.mut_or_insert_default().value = Some(TypeValue::TensorType(tensor));
}

/// The value of a one-number constant (a weight in the file, or a Constant node), whatever its float type.
fn scalar(g: &onnx_protobuf::GraphProto, name: &str) -> Option<f32> {
    let t = g.initializer.iter().find(|t| t.name == name).or_else(|| g.node.iter().find(|n| n.op_type == "Constant" && n.output[0] == name).and_then(|n| attribute(n, "value")).and_then(|a| a.t.as_ref()))?;
    let raw = &t.raw_data;
    match t.data_type {
        x if x == DataType::FLOAT16 as i32 => Some(if raw.len() >= 2 { half::f16::from_le_bytes([raw[0], raw[1]]).to_f32() } else { half::f16::from_bits(*t.int32_data.first()? as u16).to_f32() }),
        x if x == DataType::FLOAT as i32 => Some(if raw.len() >= 4 { f32::from_le_bytes(raw[..4].try_into().ok()?) } else { *t.float_data.first()? }),
        _ => None,
    }
}

/// Every spelled-out layer norm (x - mean, squared, mean, + epsilon, square root, divide, times scale, plus shift) as one
/// LayerNormalization of x / 16 with epsilon / 256 (exactly the same result, and nothing near float16's limit when it is squared).
fn layer_norms(mut model: ModelProto) -> Result<ModelProto> {
    const SHRINK: f32 = 16.0;
    let g = model.graph.as_mut().ok_or("no graph")?;
    let prod = producers(&g.node);
    let mut readers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, n) in g.node.iter().enumerate() {
        for x in &n.input {
            readers.entry(x.as_str()).or_default().push(i);
        }
    }
    let only = |t: &str, op: &str| -> Option<usize> {
        match readers.get(t).map(|r| r.as_slice()) {
            Some([i]) if g.node[*i].op_type == op => Some(*i),
            _ => None,
        }
    };
    let mut dead = HashSet::new();
    // (where the new nodes go, the nodes)
    let mut added: Vec<(usize, Vec<NodeProto>)> = Vec::new();
    for (k, pow) in g.node.iter().enumerate().filter(|(_, n)| n.op_type == "Pow").map(|(i, _)| i).enumerate() {
        let found = (|| {
            let p = &g.node[pow];
            if scalar(g, &p.input[1])? != 2.0 {
                return None;
            }
            let sub = *prod.get(&p.input[0])?;
            let (x, mean) = (&g.node[sub].input[0], &g.node[sub].input[1]);
            let mean_at = *prod.get(mean)?;
            if g.node[sub].op_type != "Sub" || g.node[mean_at].op_type != "ReduceMean" || &g.node[mean_at].input[0] != x {
                return None;
            }
            let var = only(&p.output[0], "ReduceMean")?;
            let add_eps = only(&g.node[var].output[0], "Add")?;
            let eps = scalar(g, g.node[add_eps].input.iter().find(|i| **i != g.node[var].output[0])?)?;
            let sqrt = only(&g.node[add_eps].output[0], "Sqrt")?;
            let div = only(&g.node[sqrt].output[0], "Div")?;
            if g.node[div].input[0] != g.node[sub].output[0] {
                return None;
            }
            let mul = only(&g.node[div].output[0], "Mul")?;
            let gamma = g.node[mul].input.iter().find(|i| **i != g.node[div].output[0])?.clone();
            let add = only(&g.node[mul].output[0], "Add")?;
            let beta = g.node[add].input.iter().find(|i| **i != g.node[mul].output[0])?.clone();
            // the mean of x has to be read only by this norm's Sub (it is removed with it)
            if readers.get(mean.as_str())?.len() != 1 || readers.get(g.node[sub].output[0].as_str())?.len() != 2 {
                return None;
            }
            Some((x.clone(), gamma, beta, eps, [mean_at, sub, pow, var, add_eps, sqrt, div, mul, add]))
        })();
        let Some((x, gamma, beta, eps, parts)) = found else { continue };
        let out = g.node[parts[8]].output[0].clone();
        let small = format!("ln{k}_small");
        added.push((
            parts[8],
            vec![
                node("Mul", &[&x, "ln_shrink"], &[&small], &format!("ln{k}_shrink"), vec![]),
                node("LayerNormalization", &[&small, &gamma, &beta], &[&out], &format!("ln{k}"), vec![int("axis", -1), float("epsilon", eps / (SHRINK * SHRINK)), int("stash_type", 1)]),
            ],
        ));
        dead.extend(parts);
    }
    if added.is_empty() {
        return Err("no spelled-out layer norm: this is not the file the patch is for".into());
    }
    g.initializer.push(TensorProto { name: "ln_shrink".into(), data_type: DataType::FLOAT16 as i32, raw_data: half::f16::from_f32(1.0 / SHRINK).to_le_bytes().to_vec(), ..Default::default() });
    let mut nodes = Vec::with_capacity(g.node.len());
    let mut added = added.into_iter().peekable();
    for (i, n) in std::mem::take(&mut g.node).into_iter().enumerate() {
        if !dead.contains(&i) {
            nodes.push(n);
        }
        while added.peek().is_some_and(|(at, _)| *at == i) {
            nodes.extend(added.next().unwrap().1);
        }
    }
    g.node = nodes;
    g.value_info.clear();
    for o in &mut model.opset_import {
        if o.domain.is_empty() && o.version < 17 {
            o.version = 17; // LayerNormalization is new in opset 17; nothing else in these files changed between 14 and 17
        }
    }
    Ok(model)
}

/// The convolutions get their padding written out as zeros (`pads`), in place of "VALID" (which CoreML does not take) or nothing;
/// a convolution that pads some other way is left as it is. Then the matrix products by a weight become `Gemm`s (`weights_as_gemm`).
fn siglip_coreml(mut model: ModelProto) -> Result<ModelProto> {
    let g = model.graph.as_mut().ok_or("no graph")?;
    let mut changed = 0;
    for n in g.node.iter_mut().filter(|n| n.op_type == "Conv") {
        let auto = attribute(n, "auto_pad").map_or(&b"NOTSET"[..], |a| &a.s[..]);
        if attribute(n, "pads").is_none() && (auto == b"NOTSET" || auto == b"VALID") {
            n.attribute.retain(|a| a.name != "auto_pad");
            n.attribute.push(ints("pads", &[0, 0, 0, 0]));
            changed += 1;
        }
    }
    if changed == 0 {
        return Err("no convolution without padding: this is not the file the patch is for".into());
    }
    if weights_as_gemm(g) == 0 {
        return Err("no matrix product by a weight: this is not the file the patch is for".into());
    }
    Ok(model)
}

/// Every `MatMul` of a tensor by a 2-D weight kept in the file becomes Reshape (to rows), `Gemm` with the weight turned the other
/// way round (`transB`), Reshape back. The same numbers, but ONNX Runtime's CoreML provider writes a `MatMul`'s weight into the
/// CoreML program as text (a transposed copy, every number spelled out: 1.1 GB for SigLIP 256, which CoreML parses again every
/// time the model is loaded: 48 s, and 14 minutes the first time) while a `Gemm`'s weight goes to the binary weights file as it
/// is (324 KB of program, loaded in 0.2 s, 14 s the first time). Returns how many were changed.
fn weights_as_gemm(g: &mut onnx_protobuf::GraphProto) -> usize {
    let weights: HashMap<String, usize> = g.initializer.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
    let mut count = 0;
    let mut nodes = Vec::with_capacity(g.node.len() * 2);
    for n in std::mem::take(&mut g.node) {
        let weight = (n.op_type == "MatMul").then(|| weights.get(&n.input[1]).copied()).flatten().and_then(|i| turned(&g.initializer[i]));
        let Some(turned) = weight else {
            nodes.push(n);
            continue;
        };
        let (rows, columns) = (turned.dims[1], turned.dims[0]); // turned: columns x rows of the weight
        let k = format!("mm{count}");
        let (flat, product, shape, lead, shape_lead) = (format!("{k}_flat"), format!("{k}_product"), format!("{k}_shape"), format!("{k}_lead"), format!("{k}_lead_dims"));
        let new = |name: &str, values: &[i64]| int64s(&format!("{k}_{name}"), values);
        let weight_name = format!("{}_turned", n.input[1]);
        g.initializer.extend([
            TensorProto { name: weight_name.clone(), ..turned },
            new("rows", &[-1, rows]),
            new("from", &[0]),
            new("to", &[2]),
            new("columns", &[columns]),
        ]);
        nodes.extend([
            node("Reshape", &[&n.input[0], &format!("{k}_rows")], &[&flat], &format!("{k}_to_rows"), vec![]),
            node("Gemm", &[&flat, &weight_name], &[&product], &format!("{k}_gemm"), vec![int("transB", 1)]),
            node("Shape", &[&n.input[0]], &[&shape], &format!("{k}_shape_of"), vec![]),
            node("Slice", &[&shape, &format!("{k}_from"), &format!("{k}_to")], &[&lead], &format!("{k}_lead_of"), vec![]),
            node("Concat", &[&lead, &format!("{k}_columns")], &[&shape_lead], &format!("{k}_new_shape"), vec![int("axis", 0)]),
            node("Reshape", &[&product, &shape_lead], &[&n.output[0]], &format!("{k}_back"), vec![]),
        ]);
        count += 1;
    }
    g.node = nodes;
    let used: HashSet<String> = g.node.iter().flat_map(|n| n.input.iter().cloned()).chain(g.output.iter().map(|o| o.name.clone())).collect();
    g.initializer.retain(|t| used.contains(&t.name));
    g.value_info.clear();
    count
}

/// A 2-D weight kept in the file itself (float16 or float32), turned over (rows and columns swapped).
fn turned(t: &TensorProto) -> Option<TensorProto> {
    let size = match t.data_type {
        d if d == DataType::FLOAT16 as i32 => 2,
        d if d == DataType::FLOAT as i32 => 4,
        _ => return None,
    };
    if t.dims.len() != 2 || !t.external_data.is_empty() || t.raw_data.len() as i64 != t.dims[0] * t.dims[1] * size as i64 {
        return None;
    }
    let (rows, columns) = (t.dims[0] as usize, t.dims[1] as usize);
    let mut raw = vec![0u8; t.raw_data.len()];
    for r in 0..rows {
        for c in 0..columns {
            raw[(c * rows + r) * size..][..size].copy_from_slice(&t.raw_data[(r * columns + c) * size..][..size]);
        }
    }
    Some(TensorProto { data_type: t.data_type, dims: vec![t.dims[1], t.dims[0]], raw_data: raw, ..Default::default() })
}

/// The same network computing in float32: every float16 weight, constant and type made float32 (exactly: every float16 number is
/// a float32 one). For the processor, whose float16 arithmetic is slower. The weights are written into the file itself, so a
/// network whose float32 weights would pass 2 GB (protobuf's limit) is refused.
fn to_float32(dir: &Path, mut model: ModelProto) -> Result<ModelProto> {
    let f16 = DataType::FLOAT16 as i32;
    let f32_type = DataType::FLOAT as i32;
    let widen = |bits: &[u8]| -> Vec<u8> { bits.chunks_exact(2).flat_map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32().to_le_bytes()).collect() };
    let mut external: HashMap<String, std::fs::File> = HashMap::new();
    let mut tensor = |t: &mut TensorProto| -> Result<()> {
        if t.data_type != f16 {
            return Ok(());
        }
        let bits: Vec<u8> = if !t.external_data.is_empty() {
            let get = |k: &str| t.external_data.iter().find(|e| e.key == k).map(|e| e.value.clone());
            let location = get("location").ok_or("external weights without a location")?;
            let offset: u64 = get("offset").map_or(Ok(0), |v| v.parse()).map_err(|e| format!("{e}"))?;
            let length: usize = get("length").ok_or("external weights without a length")?.parse().map_err(|e| format!("{e}"))?;
            let file = match external.entry(location.clone()) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(std::fs::File::open(dir.join(&location)).map_err(|e| format!("reading {location}: {e}"))?),
            };
            let mut buf = vec![0u8; length];
            use std::io::{Read, Seek, SeekFrom};
            file.seek(SeekFrom::Start(offset)).and_then(|_| file.read_exact(&mut buf)).map_err(|e| format!("reading {location}: {e}"))?;
            t.external_data.clear();
            t.data_location = onnx_protobuf::tensor_proto::DataLocation::DEFAULT.into();
            buf
        } else if !t.raw_data.is_empty() {
            std::mem::take(&mut t.raw_data)
        } else {
            // float16 kept as 16-bit numbers in int32_data
            std::mem::take(&mut t.int32_data).iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
        };
        t.raw_data = widen(&bits);
        t.data_type = f32_type;
        Ok(())
    };
    let g = model.graph.as_mut().ok_or("no graph")?;
    for t in &mut g.initializer {
        tensor(t)?;
    }
    for n in &mut g.node {
        for a in &mut n.attribute {
            if let Some(t) = a.t.as_mut() {
                tensor(t)?;
            }
            if (n.op_type == "Cast" && a.name == "to") || a.name == "dtype" {
                if a.i == f16 as i64 {
                    a.i = f32_type as i64;
                }
            }
        }
    }
    for v in g.input.iter_mut().chain(g.output.iter_mut()).chain(g.value_info.iter_mut()) {
        if let Some(onnx_protobuf::type_proto::Value::TensorType(t)) = v.type_.as_mut().and_then(|t| t.value.as_mut()) {
            if t.elem_type == f16 {
                t.elem_type = f32_type;
            }
        }
    }
    if model.compute_size() > i32::MAX as u64 {
        return Err("its float32 weights are over 2 GB".into());
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models;
    use std::sync::atomic::AtomicBool;

    fn ops(m: &ModelProto, op: &str) -> usize {
        m.graph.node.iter().filter(|n| n.op_type == op).count()
    }

    /// Gemma's photo encoder (its 112 KB network; the weights are not needed to patch it), downloaded into .cache/test/.
    /// cargo test --release patch -- --ignored
    #[test]
    #[ignore]
    fn gemma_patches_come_out_as_the_python_ones_did() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/patch-gemma");
        let spec = models::spec(models::GEMMA).unwrap();
        let file = spec.files.iter().find(|f| f.path == GEMMA_VISION).unwrap();
        crate::modelstore::fetch(&ureq::Agent::new_with_defaults(), spec, file, &dir, &AtomicBool::new(false), &|_| {}).unwrap();
        for name in [GEMMA_CUDA, GEMMA_TENSORRT, GEMMA_POOL] {
            let _ = std::fs::remove_file(dir.join(name));
        }

        let fused = load(&ensure(&dir, GEMMA_CUDA).unwrap()).unwrap();
        assert_eq!(ops(&fused, "MultiHeadAttention"), 16);
        assert!(fused.graph.node.iter().filter(|n| n.op_type == "MultiHeadAttention").all(|n| n.input.len() == 6 && n.input[5] == "fuse_attention_bias"));
        let original = load(&dir.join(GEMMA_VISION)).unwrap();
        assert_eq!(ops(&fused, "Pad"), ops(&original, "Pad") - 32);
        assert_eq!(fused.graph.node.len(), original.graph.node.len() - 48 + 6);

        let layers = load(&ensure(&dir, GEMMA_TENSORRT).unwrap()).unwrap();
        let pool = load(&dir.join(GEMMA_POOL)).unwrap();
        for m in [&layers, &pool] {
            assert_eq!(ops(m, "MultiHeadAttention") + ops(m, "SimplifiedLayerNormalization"), 0);
        }
        assert_eq!(ops(&layers, "RMSNormalization") + ops(&pool, "RMSNormalization"), 113); // the last norm comes after the layers
        assert_eq!(layers.graph.output.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), ["hidden"]);
        assert_eq!(pool.graph.input.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), ["hidden", "pixel_position_ids"]);
        assert_eq!(pool.graph.output.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), ["image_features"]);
        // the weights stay in the publisher's file, which both parts point at
        for m in [&layers, &pool] {
            assert!(m.graph.initializer.iter().filter(|t| !t.external_data.is_empty()).all(|t| t.external_data.iter().any(|e| e.value == "vision_encoder_fp16.onnx_data")));
        }
        eprintln!("CUDA: {} nodes; TensorRT layers: {} nodes, pooling: {} nodes", fused.graph.node.len(), layers.graph.node.len(), pool.graph.node.len());
    }

    /// How fast the processor runs SigLIP 256's photo encoder in float16 and as the float32 copy (the model has to be
    /// downloaded; the copy is made in .cache/test/ and removed). cargo test --release processor_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn processor_speed_float16_against_float32() {
        // SAFETY: set before ONNX Runtime starts, in a test run on its own
        unsafe { std::env::set_var("PHOTO_MAP_BACKEND", "cpu") };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = models::spec(models::SIGLIP_BASE_256).unwrap().dir(root);
        let half = dir.join("onnx/vision_model_fp16.onnx");
        let test = root.join(".cache/test/speed");
        std::fs::create_dir_all(&test).unwrap();
        let full = test.join("vision_model_fp32.onnx");
        save(&to_float32(&dir.join("onnx"), load(&half).unwrap()).unwrap(), &full).unwrap();
        let rt = crate::embed::Runtime::start(root).unwrap();
        let pixels: Vec<f32> = (0..4 * 3 * 256 * 256).map(|i| ((i * 7919) % 255) as f32 / 127.5 - 1.0).collect();
        let mut out = Vec::new();
        for (name, file) in [("float16", &half), ("float32", &full)] {
            let mut session = ort::session::Session::builder(&rt.env).unwrap().commit_from_file(file).unwrap();
            let mut run = || {
                let input = ort::value::Tensor::from_array(([4usize, 3, 256, 256], pixels.clone())).unwrap();
                let o = session.run(ort::inputs!["pixel_values" => input]).unwrap();
                o["pooler_output"].try_extract_tensor::<f32>().unwrap().1.to_vec()
            };
            run();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                run();
            }
            let speed = 20.0 / t.elapsed().as_secs_f32();
            eprintln!("{name}: {speed:.1} photos/s");
            out.push(run());
        }
        let c = out[0].chunks(768).zip(out[1].chunks(768)).map(|(a, b)| {
            let (d, x, y) = a.iter().zip(b).fold((0f32, 0f32, 0f32), |(d, x, y), (p, q)| (d + p * q, x + p * p, y + q * q));
            d / (x.sqrt() * y.sqrt())
        }).fold(1f32, f32::min);
        eprintln!("lowest cosine between them: {c:.5}");
        let _ = std::fs::remove_dir_all(&test);
    }
}
