"""Export CLIP ViT-B/32's vision encoder to ONNX and dump golden reference tensors.

Offline Python only. Produces everything the C++ engine is checked against:

    out/
    ├── model.onnx          # the model the C++ engine loads
    ├── manifest.json       # index of everything below, plus per-node graph listing
    ├── tensors/            # every intermediate tensor, one raw .bin per node output
    └── golden/             # end-to-end inputs and outputs
        ├── pixel_values.bin    [N, 3, 224, 224] preprocessed images (graph input)
        ├── image_embeds.bin    [N, 512]  ONNX Runtime output (graph output)
        ├── text_embeds.bin     [L, 512]  unit-length label embeddings (PyTorch)
        └── probs.bin           [N, L]    softmax(logit_scale * img_unit @ text_unit^T)

Every .bin is raw little-endian, C order (row-major), no header. dtype and shape
are in manifest.json.

The oracle is ONNX Runtime on CPU in fp32 with every graph optimisation disabled,
so each intermediate is exactly what that node produces, not a fused variant.
ONNX Runtime's output is also checked against PyTorch's encode_image, so a broken
export shows up here rather than as a mystery in the C++ engine.

The text encoder is NOT exported. The engine is image-to-image retrieval; label
embeddings are dumped only so the C++ side can reproduce test.py's label
probabilities end to end (normalise, matmul with text_embeds, softmax).
"""

import argparse
import collections
import json
import math
import re
import sys
import warnings
from pathlib import Path

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

# torch.onnx's dynamo exporter prints emoji; the default Windows console codepage
# cannot encode them and the export dies with UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

script_dir = Path(__file__).resolve().parent
clip_test_dir = script_dir.parent / "clip_test"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ViT-B/32", help="CLIP model name for clip.load")
    p.add_argument("--images", type=Path, default=clip_test_dir / "images")
    p.add_argument("--labels", type=Path, default=clip_test_dir / "labels.txt")
    p.add_argument("--out", type=Path, default=script_dir / "out")
    p.add_argument("--batch", type=int, default=0,
                   help="fixed batch size baked into the graph; images are cycled to fill it. "
                        "0 = one slot per image (default)")
    p.add_argument("--opset", type=int, default=18,
                   help=">= 17 keeps LayerNormalization as one node instead of 5-6 primitives")
    p.add_argument("--no-intermediates", action="store_true",
                   help="skip the per-node tensor dump (it is a few hundred MB)")
    return p.parse_args()


def load_images(image_dir, preprocess, batch):
    # Sorted, so the golden tensors have a stable, documented order across runs
    # and machines. test.py uses filesystem order, which is not guaranteed.
    files = sorted(f for f in image_dir.iterdir() if f.is_file())
    if not files:
        sys.exit(f"no images in {image_dir}")
    batch = batch or len(files)
    slots = [files[i % len(files)] for i in range(batch)]
    pixels = torch.stack([preprocess(Image.open(f)) for f in slots])
    return slots, pixels


def export_onnx(visual, pixels, path, opset):
    # dynamo=True: the legacy TorchScript exporter emits ~990 nodes for this
    # model, half of them runtime shape arithmetic (Shape/Gather/Cast/Mod/Sqrt).
    # The dynamo exporter resolves those statically: ~480 nodes, 14 op types.
    # No dynamic_shapes, so the batch size is fixed in the graph.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(visual, (pixels,), str(path), dynamo=True, opset_version=opset,
                          input_names=["pixel_values"], output_names=["image_embeds"],
                          external_data=False)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model


def run_with_intermediates(model, feeds, want_intermediates):
    """Run under ONNX Runtime, returning {tensor name: array} for every node output."""
    model = onnx.ModelProto.FromString(model.SerializeToString())  # don't touch the saved copy
    if want_intermediates:
        existing = {o.name for o in model.graph.output}
        for node in model.graph.node:
            for name in node.output:
                if name and name not in existing:
                    # Name-only ValueInfo: ORT fills in type and shape itself.
                    model.graph.output.append(onnx.ValueInfoProto(name=name))
                    existing.add(name)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def safe_filename(index, name):
    # Tensor names look like "/transformer/resblocks.0/ln_1/LayerNormalization_output_0".
    # The index prefix keeps files in execution order and keeps them unique after sanitising.
    return f"{index:04d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', name).strip('_')[:120]}.bin"


def save_bin(path, array):
    array = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(array.dtype.newbyteorder("<"), copy=False).tofile(path)
    return {"file": path.as_posix(), "dtype": array.dtype.name, "shape": list(array.shape)}


def attribute_value(attr):
    v = onnx.helper.get_attribute_value(attr)
    return v.decode("utf-8") if isinstance(v, bytes) else v  # STRING attributes are bytes


def initializer_checksums(model):
    """Cheap fingerprints of every weight, so the C++ parser can verify it read them correctly
    without duplicating 350 MB of weights on disk."""
    out = []
    for init in model.graph.initializer:
        a = onnx.numpy_helper.to_array(init)
        flat = a.reshape(-1)
        out.append({
            "name": init.name,
            "dtype": a.dtype.name,
            "shape": list(a.shape),
            "stored_as": "raw_data" if init.HasField("raw_data") else "typed_field",
            "sum": float(flat.astype(np.float64).sum()),
            "abs_sum": float(np.abs(flat.astype(np.float64)).sum()),
            "first": flat[:8].tolist(),
            "last": flat[-8:].tolist(),
        })
    return out


def compare(label, ours, ref):
    ours = ours.astype(np.float64)
    ref = ref.astype(np.float64)
    max_abs = float(np.max(np.abs(ours - ref)))
    cos = (ours * ref).sum(-1) / (np.linalg.norm(ours, axis=-1) * np.linalg.norm(ref, axis=-1))
    print(f"  {label}: max_abs_diff={max_abs:.3e}  min_cosine={cos.min():.8f}")
    return {"max_abs_diff": max_abs, "min_cosine": float(cos.min())}


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    args = parse_args()
    out = args.out.resolve()
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    (out / "golden").mkdir(parents=True, exist_ok=True)

    # CPU + .float(): a true fp32 reference. On CUDA clip.load gives fp16, and
    # TF32 matmuls would also make the reference lower precision than our kernels.
    model, preprocess = clip.load(args.model, device="cpu")
    model = model.float().eval()

    slots, pixels = load_images(args.images, preprocess, args.batch)
    labels = [line.rstrip("\n") for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Batch {len(slots)} from {args.images}, {len(labels)} labels")

    # --- export ---------------------------------------------------------------
    onnx_path = out / "model.onnx"
    print(f"Exporting {args.model} vision encoder, opset {args.opset} -> {onnx_path}")
    onnx_model = export_onnx(model.visual, pixels, onnx_path, args.opset)
    op_hist = collections.Counter(n.op_type for n in onnx_model.graph.node)
    print(f"  {len(onnx_model.graph.node)} nodes, {len(onnx_model.graph.initializer)} initializers")
    print("  op histogram (this is the kernel list):")
    for op, count in op_hist.most_common():
        print(f"    {op:<20} {count}")

    # --- run the oracle -------------------------------------------------------
    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    feeds = {input_name: pixels.numpy()}
    print("Running ONNX Runtime (CPU, fp32, optimisations off)")
    values = run_with_intermediates(onnx_model, feeds, not args.no_intermediates)
    image_embeds = values[output_name]

    # --- sanity: the export must match PyTorch before it is worth anything -----
    with torch.no_grad():
        torch_embeds = model.encode_image(pixels).numpy()
        text = model.encode_text(clip.tokenize(labels)).numpy()
        logit_scale = float(model.logit_scale.exp())
    print("Checks:")
    export_check = compare("ONNX Runtime vs PyTorch encode_image", image_embeds, torch_embeds)
    if export_check["min_cosine"] < 0.9999:
        sys.exit("export does not match PyTorch; do not use these goldens")

    # --- end-to-end label probabilities, same maths as test.py ------------------
    img_unit = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
    text_unit = text / np.linalg.norm(text, axis=-1, keepdims=True)
    probs = softmax(logit_scale * img_unit @ text_unit.T)
    print("Predictions:")
    for f, row in zip(slots, probs):
        i = int(row.argmax())
        print(f"  {f.name:<24} {labels[i]} ({row[i]:.2%})")

    # --- write everything -----------------------------------------------------
    rel = lambda p: p.relative_to(out)
    golden = {
        "pixel_values": save_bin(out / "golden" / "pixel_values.bin", pixels.numpy()),
        "image_embeds": save_bin(out / "golden" / "image_embeds.bin", image_embeds),
        "text_embeds": save_bin(out / "golden" / "text_embeds.bin", text_unit.astype(np.float32)),
        "probs": save_bin(out / "golden" / "probs.bin", probs.astype(np.float32)),
    }
    for g in golden.values():
        g["file"] = rel(Path(g["file"])).as_posix()

    nodes, tensors = [], {}
    # The graph input is a tensor too; node 0's inputs reference it by name.
    tensors[input_name] = dict(golden["pixel_values"], producer=None)
    for i, node in enumerate(onnx_model.graph.node):
        nodes.append({
            "index": i,
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {a.name: attribute_value(a) for a in node.attribute
                           if a.type in (onnx.AttributeProto.INT, onnx.AttributeProto.FLOAT,
                                         onnx.AttributeProto.INTS, onnx.AttributeProto.FLOATS,
                                         onnx.AttributeProto.STRING)},
        })
        for name in node.output:
            if name in values:
                entry = save_bin(out / "tensors" / safe_filename(i, name), values[name])
                entry["file"] = rel(Path(entry["file"])).as_posix()
                entry["producer"] = i
                tensors[name] = entry

    manifest = {
        "model": {"name": args.model, "file": "model.onnx", "opset": args.opset,
                  "input": input_name, "output": output_name},
        "batch": len(slots),
        "images": [str(f) for f in slots],
        "labels": labels,
        "logit_scale": logit_scale,
        "byte_order": "little",
        "layout": "row-major (C order), raw, no header",
        "oracle": f"onnxruntime {ort.__version__} CPUExecutionProvider, fp32, ORT_DISABLE_ALL",
        "exported_with": f"torch {torch.__version__}, onnx {onnx.__version__}",
        "checks": {"onnxruntime_vs_torch": export_check},
        "op_histogram": dict(op_hist.most_common()),
        "golden": golden,
        "nodes": nodes,
        "tensors": tensors,
        "initializers": initializer_checksums(onnx_model),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    total = sum(math.prod(t["shape"]) * np.dtype(t["dtype"]).itemsize for t in tensors.values())
    print(f"Wrote {len(tensors)} tensors ({total / 2**20:.0f} MiB), manifest at {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
