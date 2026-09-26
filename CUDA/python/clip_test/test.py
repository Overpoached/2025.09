import clip
import math
import torch
import time
from PIL import Image
from pathlib import Path

# number of image handled per batch
BATCH = 32

# Encoding benchmark settings. Same defaults as bench::BenchConfig in
# src/bench/harness.h, so the Python and C++ numbers are directly comparable.
WARMUP_ITERS = 25
MEASURE_ITERS = 200

# clip.load gives an fp16 model on CUDA. The C++ engine is fp32, so compare
# against fp32 unless you are deliberately measuring the fp16 path.
FP32 = True

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
file_path = script_dir / "labels.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
if FP32:
    model = model.float()
    # TF32 runs fp32 matmuls/convs on tensor cores at reduced precision. Off,
    # so this is a true fp32 baseline like our kernels.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
print("Device: ", device)

# Specify the directory path (use '.' for the current directory)
image_directory_path = script_dir / "images"

# Get all file paths in the directory and store them in a list
image_file_array = [str(file) for file in image_directory_path.iterdir() if file.is_file()]

print("Images count: ", len(image_file_array))

with open(file_path, 'r') as file:
    lines = [line.rstrip() for line in file]

print("Labels count: ", len(lines))
print("Labels:     ", lines)

print("===== Starting image identification =====")

texts = clip.tokenize(lines).to(device)

with torch.no_grad():
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for start in range(0, len(image_file_array), BATCH):
        chunk = image_file_array[start:start + BATCH]
        images = torch.stack([preprocess(Image.open(f)) for f in chunk]).to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # cosine similarity × CLIP's scale (100), same as model() does internally
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        for image_file, row in zip(chunk, probs):
            i = row.argmax()
            print("Label probs:", row)
            print(f"{image_file}:\n {lines[i]} ({row[i]:.2%})")
            print("----------------------------------------")

print("===== End image identification =====")


# Nearest-rank percentile, same definition as bench::PercentileSorted in
# src/bench/stats.cpp: always returns a sample that actually occurred.
def percentile_sorted(sorted_samples, percent):
    rank = math.ceil(percent / 100.0 * len(sorted_samples))
    return sorted_samples[min(max(rank - 1, 0), len(sorted_samples) - 1)]


def median_sorted(sorted_samples):
    n = len(sorted_samples)
    mid = n // 2
    return 0.5 * (sorted_samples[mid - 1] + sorted_samples[mid]) if n % 2 == 0 else sorted_samples[mid]


print("===== Starting encode benchmark =====")

# Preprocessing is CPU work (PIL), so wall clock is the right timer here.
# Reported separately: week 5 moves this onto the GPU.
t0 = time.perf_counter()
pixels = torch.stack([preprocess(Image.open(f)) for f in image_file_array])
preprocess_ms = (time.perf_counter() - t0) * 1000.0

# Fill a full batch by cycling the images. Pixel values do not affect timing,
# only the shape does, so a fixed [BATCH, 3, 224, 224] input is what matters.
reps = math.ceil(BATCH / len(pixels))
batch = pixels.repeat(reps, 1, 1, 1)[:BATCH].to(device)
if FP32:
    batch = batch.float()
else:
    batch = batch.to(next(model.parameters()).dtype)

samples = []
with torch.no_grad():
    for _ in range(WARMUP_ITERS):
        model.encode_image(batch)

    if device == "cuda":
        # CUDA events, one pair per iteration, like CudaEventTimer in the C++
        # harness. Only the GPU forward pass is inside the window: the input is
        # already on the device, so no host-to-device copy is counted.
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(MEASURE_ITERS):
            start.record()
            model.encode_image(batch)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
    else:
        for _ in range(MEASURE_ITERS):
            t0 = time.perf_counter()
            model.encode_image(batch)
            samples.append((time.perf_counter() - t0) * 1000.0)

samples.sort()
median = median_sorted(samples)
p25 = percentile_sorted(samples, 25.0)
p75 = percentile_sorted(samples, 75.0)
p99 = percentile_sorted(samples, 99.0)

print("GPU:          ", torch.cuda.get_device_name(0) if device == "cuda" else "n/a")
print("torch:        ", torch.__version__)
print("Precision:    ", "fp32 (TF32 off)" if FP32 else str(next(model.parameters()).dtype))
print("Input:        ", list(batch.shape))
print(f"Iterations:    warmup={WARMUP_ITERS} measured={MEASURE_ITERS}")
print(f"Preprocess:    {preprocess_ms:.3f} ms for {len(pixels)} images "
      f"({preprocess_ms / len(pixels):.3f} ms/image, CPU, wall clock)")
print(f"Encode (ms):   median={median:.3f} p99={p99:.3f} min={samples[0]:.3f} max={samples[-1]:.3f}")
print(f"robust_cv:     {(p75 - p25) / median:.2%}   tail_ratio (p99/median): {p99 / median:.2f}")
print(f"Throughput:    {BATCH / (median / 1000.0):.1f} images/sec at median")
print(f"Per image:     {median / BATCH:.3f} ms")

print("===== End encode benchmark =====")