# clip_test — CLIP ViT-B/32 reference sanity check

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

Runs OpenAI's CLIP ViT-B/32 in PyTorch over every image in `images/` and, for
each one, picks the best-matching label from `labels.txt`. Images are processed
in batches. Offline Python only. This is the **reference** the CUDA engine is
checked against, not part of the engine.

## Folder layout

```
clip_test/
├── test.py          # the script
├── labels.txt       # candidate labels, one per line
├── images/          # every file in here gets classified
│   ├── cat.jpg
│   ├── dog.jpg
│   └── ...
└── README.md
```

Both paths are resolved **relative to `test.py`**, not to the folder you run
it from. So you can run the script from anywhere.

## Using the script

### Images: `images/`

- Drop image files (`.jpg`, `.png`, ...) directly into `images/`.
- **Every file** in the folder is treated as an image. A non-image file (e.g.
  `notes.txt`, `desktop.ini`, `Thumbs.db`) makes `Image.open` crash, so keep the
  folder clean.
- Subfolders are ignored (not searched).
- Any size and aspect ratio is fine. `preprocess` resizes the shorter side to
  224, centre-crops to 224×224 and normalises.
- Files are processed in whatever order the filesystem lists them (not
  guaranteed alphabetical).

### Labels: `labels.txt`

One label per line, plain text:

```
a dog
a cat
a car
```

- **No blank lines**, including at the end of the file. A blank line becomes an
  empty-string label that still competes for probability.
- Phrase labels like a caption: `a photo of a cat` usually beats a bare `cat`.
  CLIP was trained on image captions.
- **The script always picks one of your labels, even if none fit.** Softmax
  turns the scores into probabilities that sum to 100% *across your labels
  only*. A photo of a bike with labels `dog / cat / car` will still report one
  of those, probably `a car`, possibly with high confidence. Add labels such as
  `a bicycle` if you want the script to be able to say so.

### Settings in `test.py`

| Setting | Default | Meaning |
|---|---|---|
| `BATCH` | `32` | Images per forward pass. Bigger uses more memory. Past the point where the GPU is saturated it adds no throughput. |
| `device` | `"cuda"` if available, else `"cpu"` | Where the model runs. Auto-detected; printed at startup. Hardcode `"cpu"` to force fp32 (see below). |

#### CPU vs GPU

- `clip.load(..., device=device)` moves only the **model weights**. Every input
  tensor must be moved too, with `.to(device)`. The script does this for
  `texts` and `images`. PyTorch never copies between host and device
  implicitly. Forgetting one gives:
  `Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0`.
- Results are copied back with `.cpu()` before `.numpy()`, because numpy only
  reads host memory.
- **Precision differs.** On `"cuda"`, `clip.load` converts the model to
  **fp16**. On `"cpu"` it stays **fp32**. Labels and probabilities agree to a
  few decimals, but raw embeddings differ slightly. **When using this script as
  the correctness oracle for the CUDA engine, compare fp32 to fp32.** Either
  force `device = "cpu"`, or call `model = model.float()` after `clip.load`.

### Output

The run starts with `Device:  cuda` (or `cpu`). Then, for each image:

```
Label probs: [0.0101 0.989 0.00079]
C:\...\clip_test\images\cat.jpg:
 a cat (98.90190%)
----------------------------------------
```

`Label probs` lists one probability per label, in `labels.txt` order, summing
to 1. The line below it is the highest-scoring label.

### How it works

1. Encode all labels **once** → `[num_labels, 512]` text embeddings.
2. For each chunk of up to `BATCH` images: preprocess, `torch.stack` into one
   `[n, 3, 224, 224]` tensor, encode → `[n, 512]` image embeddings.
3. Normalise both to unit length. Then `100 × image @ textᵀ` gives cosine
   similarity scores `[n, num_labels]`, and softmax per row turns them into
   probabilities. This is what `model(images, texts)` does internally.

## What matters for the project

| In `test.py` | In the CUDA engine |
|---|---|
| `preprocess(...)` → `[3, 224, 224]` per image, stacked to `[N, 3, 224, 224]` | Week 5 preprocessing kernels must reproduce this |
| `model.encode_image(images)` → `[N, 512]` embeddings | Weeks 6–8 ONNX executor + kernels must reproduce this |
| `BATCH` chunking | Batch-size sweep in benchmarks; dynamic batching in the server (week 11) |
| `clip.tokenize` / `encode_text` | **Not used** — the project is image-to-image retrieval |
| softmax over text matches | Replaced by nearest-neighbour search over catalog embeddings (week 10) |

## One-time setup (Windows, cmd)

https://medium.com/thedeephub/how-to-install-and-use-clip-a-complete-step-by-step-guide-99371e841ee8

### 1. Install Python 3.12 from python.org

```bat
winget install -e --id Python.Python.3.12
```

Open a **new** cmd window afterwards so `py -3.12` is on PATH.

> **Do not build the venv from Anaconda's Python.** Anaconda ships an old
> MSVC runtime (`vcruntime140.dll` / `msvcp140.dll` 14.27) in its install folder,
> which loads before the system copy. Current torch then fails with:
> `OSError: [WinError 1114] ... Error loading "...\torch\lib\c10.dll"`.

### 2. Install / update the MSVC runtime

```bat
winget install -e --id Microsoft.VCRedist.2015+.x64
```

### 3. Clone the CLIP repo

```bat
cd C:\_stuffs\repo
git clone https://github.com/openai/CLIP.git
cd CLIP
```

### 4. Create and activate the venv

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Check you are on the right interpreter — the first line must be inside `.venv`:

```bat
where python
```

### 5. Install PyTorch — CUDA build

Plain `pip install torch` on Windows gives the **CPU-only** build (`2.x.x+cpu`).
Get the exact command from <https://pytorch.org/get-started/locally/>
(Stable / Windows / Pip / Python / CUDA 12.x). It looks like:

```bat
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If you already have the CPU build, remove it first:

```bat
pip uninstall -y torch torchvision
```

### 6. Install CLIP's dependencies and CLIP itself

```bat
pip install -r requirements.txt
pip install packaging pillow
pip install -e C:\_stuffs\repo\CLIP
```

- `packaging` is imported by `clip/clip.py` but missing from `requirements.txt`.
- `pip install -e` makes `import clip` work from any folder, not just from
  inside the CLIP repo.

### 7. Verify

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import clip; print('CLIP is installed!')"
```

Expect `2.x.x+cuXXX True NVIDIA GeForce RTX 4060`. If it says `+cpu False`,
redo step 5.

## Running (every time)

```bat
cd C:\_stuffs\repo\CLIP
.venv\Scripts\activate
python C:\_stuffs\repo\2025.09\CUDA\python\clip_test\test.py
```

The prompt must show `(.venv)`. Without it, `python` may be Anaconda's and
you'll get the `c10.dll` error again.

First run downloads the ViT-B/32 weights (~338 MB) to
`%USERPROFILE%\.cache\clip\`. Later runs use the cache.

See [Using the script](#using-the-script) for where the images and labels go.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `'pip' is not recognized` | venv not activated / Python not on PATH | `.venv\Scripts\activate`, or use `py -m pip` |
| `WinError 1114 ... c10.dll` | venv built from Anaconda's Python (old MSVC runtime) | Recreate venv with `py -3.12` (step 1, 4) |
| `No module named 'packaging'` | Missing from CLIP's requirements | `pip install packaging` |
| `No module named 'clip'` | CLIP not installed into the venv | `pip install -e C:\_stuffs\repo\CLIP` |
| `can't find '__main__' module` | Passed a folder to `python` | Pass the `.py` file |
| `Can't call numpy() on Tensor that requires grad` | Model call outside `torch.no_grad()` | Keep inference inside the `with torch.no_grad():` block |
| `torch.cuda.is_available()` is `False` / script prints `Device:  cpu` | CPU-only torch build | Step 5 |
| `Expected all tensors to be on the same device ... cpu ... cuda:0` | Model on GPU, an input tensor still on CPU | Add `.to(device)` to the input tensor |
| Embeddings differ slightly between CPU and GPU runs | GPU model is fp16, CPU model is fp32 | Expected. Use fp32 for oracle comparisons (see [CPU vs GPU](#cpu-vs-gpu)) |
| `PIL.UnidentifiedImageError: cannot identify image file` | Non-image file in `images/` | Remove it (watch for `Thumbs.db`, `desktop.ini`) |
| `FileNotFoundError: ... labels.txt` or `... images` | Missing next to `test.py` | Create them, see [Folder layout](#folder-layout) |
| `The truth value of an array ... is ambiguous` | Iterating a 2D `probs` array (`[images, labels]`) as if it were 1D | Index a row first: `probs[0]`, or loop `for row in probs` |
| Every image gets a confident but wrong label | None of the labels fit, and softmax still has to pick one | Add more labels |
| Garbled file path | Backslashes in a normal string (`\r`, `\202`) | Use raw strings: `r"C:\..."` |
| `FutureWarning: torch.jit.load is deprecated` | CLIP repo uses an old API | Harmless, ignore |
