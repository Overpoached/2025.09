# python/dump_golden.py  —  usage: python dump_golden.py model.onnx golden/
import os, re, sys
import numpy as np
import onnx
import onnxruntime as ort

model_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
g = model.graph
existing = {o.name for o in g.output}
value_info = {v.name: v for v in g.value_info}
for node in g.node:
    for name in node.output:
        if name and name not in existing:
            g.output.append(value_info.get(name) or onnx.ValueInfoProto(name=name))
            existing.add(name)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep the original nodes, no fusion
sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])

inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]   # symbolic dims (e.g. 'batch') -> 1
print("input", inp.name, inp.shape, "->", shape)
x = np.random.default_rng(0).standard_normal(shape, dtype=np.float32)

manifest = []
def dump(name, arr):
    arr = np.ascontiguousarray(arr)
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".bin"   # names like /conv1/Conv_output_0
    arr.tofile(os.path.join(out_dir, fname))                  # raw little-endian, row-major
    manifest.append(f"{name}\t{fname}\t{arr.dtype}\t{','.join(map(str, arr.shape))}")

dump(inp.name, x)
names = [o.name for o in sess.get_outputs()]
for name, val in zip(names, sess.run(names, {inp.name: x})):
    dump(name, val)

with open(os.path.join(out_dir, "manifest.tsv"), "w") as f:
    f.write("\n".join(manifest) + "\n")