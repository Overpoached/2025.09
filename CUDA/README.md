# A from-scratch CUDA inference engine

Hand-written CUDA kernels and a hand-written ONNX graph executor, demonstrated
on fine-grained vehicle identification. No ONNX Runtime, no cuBLAS, no TensorRT
anywhere in the path that serves a request — those are linked into separate
binaries and used only as a correctness oracle and a benchmark opponent.

**Status: prototype v0.** The CLIP ViT-B/32 vision encoder runs end to end on
our own kernels, from JPEG bytes to a 512-dim embedding. Embeddings can be
stored, updated, deleted and counted in a local file store, and new images are
identified by k-nearest-neighbour search on the GPU. The matmul is a first
tiled version; the optimisation progression against cuBLAS is still to come.

## What it does

Store reference images under labels, then ask which label a new image is
closest to:

```bash
x64\Release\cuda.exe add   --storage data --images photos\catalog\
x64\Release\cuda.exe infer --storage data --images photos\query.jpg
```

This is **retrieval, not classification**. A label is just rows in the store,
so a new vehicle model is added by inserting embeddings, with no retraining. A
fixed N-way classifier head would have to be retrained to learn class N+1.

```
image file
  │  OpenCV decode (CPU)
  │  bicubic resize + centre crop + normalise      our CUDA kernels
  ▼
[N, 3, 224, 224] float32
  │  ONNX graph executor walks ~480 nodes          our C++
  │  Conv, MatMul, LayerNorm, Softmax, ...         our CUDA kernels
  ▼
[N, 512] embedding ──► L2-normalise ──► file store (add / update)
                                    └─► GPU brute-force kNN (infer)
                                          [Q,512] × [N,512]ᵀ on our GEMM
                                          → similarity-weighted label vote
```

## The engine

### Graph executor (`src/engine/`)

Parses the `.onnx` protobuf, loads every initializer (weight) to the GPU, and
walks the nodes in order, dispatching each one to a hand-written op. Shape
arithmetic inputs (the operands of `Reshape`, `Unsqueeze` and friends) are kept
on the host; everything else lives on the device.

It supports exactly the ops this model needs and nothing else. Any other op
fails loudly with `Unsupported operation: <op>`. This is deliberately not a
general ONNX runtime.

| Op | Notes |
|---|---|
| `Conv` | patch embedding; general padding and strides |
| `MatMul`, `Gemm` | numpy-style batching and broadcasting, `transA`/`transB`, bias |
| `Add`, `Mul` | numpy broadcasting |
| `Sigmoid` | for CLIP's QuickGELU, `x · sigmoid(1.702x)` |
| `LayerNormalization` | |
| `Softmax` | any axis |
| `Gather` | negative and scalar indices |
| `Transpose`, `Reshape`, `Concat`, `Squeeze`, `Unsqueeze` | |

The model is exported with PyTorch's dynamo exporter, which gives ~480 nodes and
14 op types (the legacy exporter produces ~990, half of them runtime shape
arithmetic). The batch size is baked into the graph; partial batches are padded
and the padding rows' outputs are discarded.

### Preprocessing (`src/engine/preprocess.cpp`, `src/kernels/preprocess/`)

Matches CLIP's Python pipeline exactly. Get this wrong and the output is quietly
garbage: the shapes still line up.

- decode on the CPU with OpenCV, ignoring EXIF orientation (as PIL does)
- resize the short side to 224 with **Pillow's bicubic filter** (a = −0.5),
  then centre-crop to 224×224, using torchvision's rounding rules
- scale to 0–1, normalise per channel with CLIP's mean and std, write RGB NCHW

The resize runs as two separable GPU passes (rows, then columns fused with
normalisation). The filter tables are built on the host, and the row pass only
covers the source rows that the crop actually reads.

### Correctness

Two oracles:

- **Per-node golden check** (`golden` command). The export script runs the model
  through ONNX Runtime with all graph optimisations disabled and dumps every
  node's output. The engine runs the same input and compares after **every
  node** (atol 1e-4, rtol 1e-3), so the first mismatch points at the faulty op.
- **Unit tests** (`test` command): kernel tests including the exact CLIP matmul
  shapes (QKV projection, attention scores and values, MLP expand and contract),
  conv cases taken from the ONNX operator docs, gather edge cases, the GPU kNN
  search checked against a CPU brute force, and the storage layer.

## Commands

All commands take `--model PATH` (default `./model/model.onnx`) and
`--storage DIR` (default `./data`) where relevant.

| Command | What it does |
|---|---|
| `add [--label L] --images A dir\ ...` | Embed images and store them. Files already stored are skipped. |
| `update [--label L] --images A dir\ ...` | Replace everything stored under the labels these images resolve to. |
| `delete { --label L ... \| --all }` | Delete everything under the given labels, or clear the store. |
| `read [--label L ...]` | Count stored embeddings per label (all labels if none given). |
| `infer [--k N] --images A dir\ ...` | Identify each image against the store. |
| `compare --images A B ...` | Embed images and print their pairwise cosine similarity. |
| `golden --golden_dir DIR` | Run the graph against ONNX Runtime's per-node reference outputs. |
| `test` | Run the unit tests. |
| `bench`, `info`, `demo` | Bandwidth benchmark, device and clock state, sanity demo. |

**Images and labels.** `--images` takes files and directories. A directory is
searched recursively for `.jpg .jpeg .jpe .png .bmp`, which is what the OpenCV
build decodes. Labels for `add` and `update`:

| `--label` | Label of each image |
|---|---|
| not given | the name of its parent directory: `cars\honda_civic_2012\001.jpg` → `honda_civic_2012` |
| once | that label, for every image |
| once per `--images` entry | that entry's label, for every image under it |

A dataset laid out one directory per class, like Stanford Cars, loads with a
single command.

**`update` only touches its own labels.** The labels the input images resolve
to define the scope. In one all-or-nothing transaction, everything stored under
those labels is deleted and the new images are stored. A file that is already
stored under a label *outside* the scope is skipped with a warning, never moved.
To move a file to another label, delete the old label first.

`--user NAME` sets `create_by` on stored rows (default: the OS user name).
`read` and `delete` do not load the model, so they return immediately.

### `infer`

Each query image is compared against **every** stored embedding, not one per
label. Its `k` nearest neighbours (default 5) each vote for their own label,
weighted by similarity. The label with the highest total wins; ties go to the
label with the nearer single neighbour. A weighted vote means one mislabelled
reference photo cannot outvote several close ones, and it still uses how close
each neighbour is.

Both stored and query embeddings are L2-normalised, so cosine similarity is a
dot product. Scoring a batch of Q queries against N stored embeddings is
therefore one `[Q, 512] × [N, 512]ᵀ` GEMM on our own matmul kernel. The stored
matrix is uploaded to the GPU once, at start-up.

For each image, the output gives the best label, its share of the vote, how many
of the k neighbours it got, and its nearest distance (`1 − cosine similarity`).
Below that are every candidate label and the raw neighbours. A summary table
follows.

## Storage

The store is accessed through a `StorageProxy` interface. The only backend today
is a local file store; a Postgres backend will implement the same interface.
Business rules live above it in `EmbedStore`, so they do not change when the
backend does:

- a file is stored at most once; paths are canonicalised before comparing
  (absolute, normalised, and lower-cased on Windows)
- an embedding and its save entry are written and deleted together
- embeddings are stored L2-normalised

| Table | Columns | File |
|---|---|---|
| save entry (duplicate check) | `id, file_path, embed_id, create_on, create_by` | `save_entry.tsv`, text |
| main data | `id, label, embed, create_on, create_by` | `main_data.bin`, binary with an `OEMB` header and version |

The file backend loads both tables into memory on open. Every change runs inside
a transaction with commit and rollback. The backend also enforces what a
database would:

- `file_path` is unique
- `embed_id` must point at an existing embedding
- every embedding has the same dimension
- an embedding cannot be deleted while a save entry still points at it

A transaction that only inserts appends to the files. One that deletes rewrites
both files through a temp file and a rename. The remaining crash windows are
documented in `file_storage_proxy.h`; anything they leave behind is reported as
a warning on the next open. Labels can repeat: several photos of one model are
several rows.

## Why the harness came first

Every performance claim this project makes has to survive being checked. That
only works if the measurement infrastructure predates the thing being measured,
so it was built before any application kernel. It enforces, by construction:

- CUDA event timing, never host wall clock
- warmup iterations discarded before anything is recorded
- a distribution over N iterations — median, p99, and dispersion — never a
  single number
- GPU clock, temperature and NVML clock-event reasons captured either side of
  every measured run and written into the log entry
- correctness verified against the expected result *before* a kernel is timed
- a non-zero exit code, and a printed reason, when the measurement environment
  was not sound enough to trust the numbers

## Hardware

RTX 4060, Ada Lovelace, sm_89. 24 SMs, 8 GB GDDR6, 24 MiB L2, 128-bit bus at
8501 MHz — **272.0 GB/s theoretical bandwidth**. The card is compute-rich
relative to its bandwidth, so memory-bound kernels are severely memory-bound
here and coalescing, tiling and fusion produce large, legible speedups.

## Building

Windows, Visual Studio 2026 (toolset v145), CUDA 13.3. Dependencies (`onnx` for
the protobuf model format, `opencv4` with only the `jpeg` and `png` features for
image decoding) come from vcpkg in manifest mode. Open `cuda.sln` at the
repository root, or from the command line:

```bash
msbuild cuda.sln /p:Configuration=Release /p:Platform=x64
```

Release builds target `sm_89` with `-lineinfo` (for Nsight Compute source
correlation) and without fast-math, so kernel arithmetic stays comparable to the
correctness oracle. **Never benchmark a Debug build** — device code is compiled
with `-G`, and the runner will refuse to call the results trustworthy.

## Getting the model

Exporting the model is offline Python work; see
[python/export/README.md](python/export/README.md). In short:

```bash
python CUDA\python\export\export_clip.py
```

This writes `model.onnx` (the graph and weights) plus the per-node reference
tensors. Point `--model` at the `.onnx` file. To prepare the reference tensors
for the `golden` command:

```bash
python CUDA\python\dump_golden\dump_golden.py <model.onnx> <out_dir>
```

Then run `cuda.exe golden --model <model.onnx> --golden_dir <out_dir>`.

## Running the benchmarks

```bash
x64\Release\cuda.exe info
x64\Release\cuda.exe bench
x64\Release\cuda.exe bench --repeats 20 --markdown CUDA\BENCHMARKS.md --note "what changed"
```

`bench` sweeps `vector_add` across problem sizes, in a scalar and a `float4`
variant. Both do one add per three memory accesses — an arithmetic intensity of
1/12 FLOP per byte — so they measure bandwidth and nothing else.

Before recording anything, lock the GPU clocks from an **elevated** prompt:

```bash
powershell -ExecutionPolicy Bypass -File CUDA\tools\lock_clocks.ps1 -Lock -GraphicsClock 2100
```

and reset them afterwards with `-Reset`. Whatever that script reports goes into
the log entry, including failure: on a consumer GeForce the memory clock
usually cannot be pinned at all, and an entry saying so is worth more than a
number with no stated clock state.

## What the baseline shows

At 64Mi elements (768 MiB moved per iteration, far beyond the 24 MiB L2) the
kernel reaches roughly 245 GB/s, about 90% of theoretical. The `float4` variant
is not meaningfully faster, which is the expected result: the scalar version
already saturates DRAM, so issuing a quarter as many memory instructions buys
nothing. At 1Mi elements the same kernel "achieves" 1000+ GB/s — the working set
fits in L2 and it is not touching DRAM at all. The runner marks those rows
rather than reporting a percentage of peak that means nothing.

Measured numbers, with methodology and hardware state, live in
[BENCHMARKS.md](BENCHMARKS.md). That file is append-only.

## Layout

```
src/bench/                measurement harness: event timing, statistics,
                          NVML clock state, reporting
src/kernels/memory/       vector_add, the bandwidth probe
src/kernels/nn/           conv, matmul, layernorm, softmax, gather, transpose,
                          elementwise
src/kernels/preprocess/   bicubic resize + normalise
src/engine/               ONNX loading, graph walk, op dispatch, tensors,
                          preprocessing, image file discovery
src/engine/ops/           one file per op: attribute parsing, shapes, kernel launch
src/engine/instances/     one class per command: add/update/delete/read (crud),
                          infer, compare, golden
src/search/               embedding store (proxy interface + file backend),
                          GPU brute-force kNN, label voting
src/test/                 unit tests
python/                   offline only: ONNX export, golden dumps, CLIP reference
tools/                    clock locking and profiling scripts
BENCHMARKS.md             append-only measurement log
```

Kernel versions are kept, never replaced. `v1_scalar` and `v2_float4` both
remain compiled and benchmarkable side by side, and the same will hold for the
matmul progression. The progression is the deliverable.

## Scope and limits

- **One model.** Only the ops CLIP ViT-B/32's vision encoder uses. Anything else
  fails loudly.
- **Fixed batch.** The batch size is set at export time.
- **Search is exact brute force.** That is fine for thousands of embeddings; an
  index comes later, with a recall-versus-latency curve.
- **The file store is single-process** and not crash-proof in every window. It
  stands in for Postgres.
- **Ids can be reused.** They come from the highest stored id + 1 when the store
  is opened, so deleting the newest rows frees their ids. Postgres sequences
  will not do this.
