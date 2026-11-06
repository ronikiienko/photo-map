//! Patches: copies of a model's networks changed for one runtime, made on this computer from the files its publisher has (so no
//! changed model file is ever hosted anywhere). Each is made the first time it is needed, when the model is loaded, and kept next
//! to the downloaded files; removing the model removes them too. The weights are not copied: a patched network points at the same
//! `.onnx_data` file as the one it was made from.
//!
//! - SigLIP on TensorRT (`SIGLIP_TENSORRT`): the publisher's file (opset 14) spells every layer norm out (mean, subtract, square,
//!   ...); TensorRT squares in float16, which overflows, and the vectors come out wrong. Each becomes one LayerNormalization, its
//!   input divided by 16 first (and epsilon by 256: the same result).
//! - On the processor (`SIGLIP_PROCESSOR`): the photo encoder in float32 (`to_float32`). ONNX Runtime's float16 arithmetic on a processor is slower: SigLIP 256 made 6.7 photos/s in float16
//!   and 8.3 in float32 on a Ryzen 5 5600H (the same vectors). Search words stay float16.
//! - SigLIP's text encoder without its token table (`SIGLIP_TEXT_ROWS`): the table (one row of numbers per token, 256k tokens:
//!   375 MB of SigLIP Base's 565) would sit on the card whole for the few rows a search reads. The patched network takes the rows
//!   instead (siglip.rs reads them from the publisher's file, mapped, not loaded); its other weights point into the publisher's
//!   `.onnx` itself (`weight_ranges`), so nothing is copied.
//! - SigLIP on CoreML (`SIGLIP_COREML`): the first layer's convolution gets its padding written out (zeros, which is what no padding
//!   means): ONNX Runtime's CoreML provider refuses a convolution without it. And every matrix product by a weight becomes a `Gemm`
//!   (`weights_as_gemm`): the provider writes a product's weight into CoreML's program as text, which CoreML parses at every load.

use onnx_protobuf::{AttributeProto, Message, ModelProto, NodeProto, StringStringEntryProto, TensorProto, attribute_proto::AttributeType, tensor_proto::DataType};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

pub const SIGLIP_COREML: &str = "onnx/vision_model_fp16.coreml.onnx";
pub const SIGLIP_TENSORRT: &str = "onnx/vision_model_fp16.tensorrt.onnx";
pub const SIGLIP_PROCESSOR: &str = "onnx/vision_model_fp16.processor.onnx";
pub const SIGLIP_TEXT_ROWS: &str = "onnx/text_model_fp16.rows.onnx";
/// The publisher's text encoder, which SIGLIP_TEXT_ROWS is made from and points into.
pub const SIGLIP_TEXT: &str = "onnx/text_model_fp16.onnx";
/// The input of SIGLIP_TEXT_ROWS: the tokens' rows, float32, batch x tokens x width.
pub const ROWS_INPUT: &str = "rows";
/// The metadata key of SIGLIP_TEXT_ROWS naming the table its rows come from (a weight of SIGLIP_TEXT).
pub const ROWS_TABLE: &str = "rows_table";

/// The patched network for `name` in a model's folder, made now if it is not there yet.
pub fn ensure(dir: &Path, name: &str) -> Result<PathBuf> {
    let path = dir.join(name);
    if path.exists() {
        return Ok(path);
    }
    match name {
        SIGLIP_COREML => save(&siglip_coreml(load(&dir.join("onnx/vision_model_fp16.onnx"))?)?, &path)?,
        SIGLIP_TENSORRT => save(&layer_norms(load(&dir.join("onnx/vision_model_fp16.onnx"))?)?, &path)?,
        SIGLIP_PROCESSOR => save(&to_float32(&dir.join("onnx"), load(&dir.join("onnx/vision_model_fp16.onnx"))?)?, &path)?,
        SIGLIP_TEXT_ROWS => {
            let source = dir.join(SIGLIP_TEXT);
            let bytes = std::fs::read(&source).map_err(|e| format!("reading {}: {e}", source.display()))?;
            let weights = weight_ranges(&bytes)?;
            let model = ModelProto::parse_from_bytes(&bytes).map_err(|e| format!("reading {}: {e}", source.display()))?;
            drop(bytes);
            let location = source.file_name().unwrap_or_default().to_string_lossy().into_owned();
            save(&text_rows(model, &weights, &location)?, &path)?
        }
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

pub fn load(path: &Path) -> Result<ModelProto> {
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

/// Which node makes each tensor.
fn producers(nodes: &[NodeProto]) -> HashMap<String, usize> {
    nodes.iter().enumerate().flat_map(|(i, n)| n.output.iter().map(move |o| (o.clone(), i))).collect()
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

/// Where each weight's numbers are inside an `.onnx` file: its name and the bytes of its raw_data. Found by walking the protobuf's
/// fields (model 7: graph, graph 5: initializer, tensor 8: name, 9: raw_data) without reading the numbers themselves.
pub fn weight_ranges(bytes: &[u8]) -> Result<HashMap<String, Range<usize>>> {
    fn varint(b: &[u8], at: &mut usize) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let x = *b.get(*at)?;
            *at += 1;
            v |= ((x & 0x7f) as u64) << shift;
            if x < 0x80 {
                return Some(v);
            }
        }
        None
    }
    // a message's length-delimited fields: (field number, its bytes); numbers and fixed-size fields are skipped
    fn fields(b: &[u8], range: Range<usize>) -> Option<Vec<(u64, Range<usize>)>> {
        let mut at = range.start;
        let mut out = Vec::new();
        while at < range.end {
            let tag = varint(b, &mut at)?;
            match tag & 7 {
                0 => drop(varint(b, &mut at)?),
                1 => at += 8,
                2 => {
                    let len = varint(b, &mut at)? as usize;
                    out.push((tag >> 3, at..at.checked_add(len)?));
                    at += len;
                }
                5 => at += 4,
                _ => return None,
            }
        }
        (at == range.end).then_some(out)
    }
    let broken = || "not an ONNX file this program can read".to_string();
    let model = fields(bytes, 0..bytes.len()).ok_or_else(broken)?;
    let graph = model.into_iter().filter(|(f, _)| *f == 7).last().ok_or_else(broken)?.1;
    let mut out = HashMap::new();
    for (_, tensor) in fields(bytes, graph).ok_or_else(broken)?.into_iter().filter(|(f, _)| *f == 5) {
        let parts = fields(bytes, tensor).ok_or_else(broken)?;
        let name = parts.iter().find(|(f, _)| *f == 8).map(|(_, r)| String::from_utf8_lossy(&bytes[r.clone()]).into_owned());
        if let (Some(name), Some((_, raw))) = (name, parts.into_iter().find(|(f, _)| *f == 9)) {
            out.insert(name, raw);
        }
    }
    Ok(out)
}

/// The text encoder fed the tokens' rows (float32, made the table's type inside) in place of token numbers: the lookup in the table
/// goes, and so does the table; what read the token numbers' shape reads the rows' first two sizes. The other weights point at
/// their bytes in `location` (`weights`: weight_ranges of that file). The table's name goes in the metadata (ROWS_TABLE).
fn text_rows(mut model: ModelProto, weights: &HashMap<String, Range<usize>>, location: &str) -> Result<ModelProto> {
    let not_it = || "no token lookup on the input: this is not the file the patch is for".to_string();
    let g = model.graph.as_mut().ok_or("no graph")?;
    let ids = g.input.iter().position(|v| v.name == "input_ids").ok_or_else(not_it)?;
    let lookup = g
        .node
        .iter()
        .position(|n| n.op_type == "Gather" && n.input.get(1).is_some_and(|i| *i == g.input[ids].name) && g.initializer.iter().any(|t| t.name == n.input[0] && t.dims.len() == 2))
        .ok_or_else(not_it)?;
    let lookup = g.node.remove(lookup);
    let table = g.initializer.iter().find(|t| t.name == lookup.input[0]).ok_or_else(not_it)?;
    let (table_name, table_type, width) = (table.name.clone(), table.data_type, table.dims[1]);
    // the new input: batch x tokens (as the token numbers were) x width, float32
    let mut rows = g.input[ids].clone();
    rows.name = ROWS_INPUT.into();
    let Some(onnx_protobuf::type_proto::Value::TensorType(t)) = rows.type_.as_mut().and_then(|t| t.value.as_mut()) else { return Err(not_it()) };
    t.elem_type = DataType::FLOAT as i32;
    let mut d = onnx_protobuf::tensor_shape_proto::Dimension::new();
    d.set_dim_value(width);
    t.shape.mut_or_insert_default().dim.push(d);
    let ids_name = std::mem::replace(&mut g.input[ids], rows).name;
    let mut nodes = vec![node("Cast", &[ROWS_INPUT], &[&lookup.output[0]], "rows_cast", vec![int("to", table_type as i64)])];
    for n in std::mem::take(&mut g.node) {
        if !n.input.contains(&ids_name) {
            nodes.push(n);
        } else if n.op_type == "Shape" {
            let k = format!("{}_rows", n.name);
            nodes.push(node("Shape", &[ROWS_INPUT], &[&format!("{k}_all")], &format!("{k}_all"), vec![]));
            nodes.push(node("Slice", &[&format!("{k}_all"), &format!("{k}_from"), &format!("{k}_to")], &[&n.output[0]], &format!("{k}_slice"), vec![]));
            g.initializer.extend([int64s(&format!("{k}_from"), &[0]), int64s(&format!("{k}_to"), &[2])]);
        } else {
            return Err(format!("the token numbers are read by a {} too: this is not the file the patch is for", n.op_type));
        }
    }
    g.node = nodes;
    g.initializer.retain(|t| t.name != table_name);
    g.value_info.clear();
    for t in &mut g.initializer {
        let Some(r) = weights.get(&t.name).filter(|r| r.len() >= 1024 && r.len() == t.raw_data.len()) else { continue };
        t.raw_data.clear();
        t.data_location = onnx_protobuf::tensor_proto::DataLocation::EXTERNAL.into();
        t.external_data = [("location", location.to_string()), ("offset", r.start.to_string()), ("length", r.len().to_string())]
            .into_iter()
            .map(|(k, v)| StringStringEntryProto { key: k.into(), value: v, ..Default::default() })
            .collect();
    }
    model.metadata_props.push(StringStringEntryProto { key: ROWS_TABLE.into(), value: table_name, ..Default::default() });
    Ok(model)
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

    /// weight_ranges finds each weight's bytes in a written model.
    #[test]
    fn weight_ranges_find_the_numbers() {
        let mut model = ModelProto::new();
        let g = model.graph.mut_or_insert_default();
        g.node.push(node("Add", &["a", "b"], &["c"], "add", vec![]));
        g.initializer.push(TensorProto { name: "a".into(), data_type: DataType::FLOAT as i32, dims: vec![3], raw_data: vec![1; 12], ..Default::default() });
        g.initializer.push(TensorProto { name: "b".into(), data_type: DataType::FLOAT as i32, dims: vec![300], raw_data: (0..1200).map(|i| i as u8).collect(), ..Default::default() });
        model.metadata_props.push(StringStringEntryProto { key: "k".into(), value: "v".into(), ..Default::default() });
        let bytes = model.write_to_bytes().unwrap();
        let found = weight_ranges(&bytes).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(&bytes[found["a"].clone()], &[1u8; 12][..]);
        assert_eq!(bytes[found["b"].clone()].to_vec(), (0..1200).map(|i| i as u8).collect::<Vec<u8>>());
    }

    /// How fast the processor runs SigLIP 256's photo encoder in float16 and as the float32 copy (the model has to be
    /// downloaded; the copy is made in .cache/test/ and removed). cargo test --release processor_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn processor_speed_float16_against_float32() {
        // SAFETY: set before ONNX Runtime starts, in a test run on its own
        unsafe { std::env::set_var("PHOTO_MAP_BACKEND", "cpu") };
        let root = &crate::data_root();
        let dir = models::spec(models::SIGLIP_BASE_256).unwrap().dir(root);
        let half = dir.join("onnx/vision_model_fp16.onnx");
        let test = root.join(".cache/test/speed");
        std::fs::create_dir_all(&test).unwrap();
        let full = test.join("vision_model_fp32.onnx");
        save(&to_float32(&dir.join("onnx"), load(&half).unwrap()).unwrap(), &full).unwrap();
        let rt = crate::embed::Runtime::shared(root).unwrap();
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
