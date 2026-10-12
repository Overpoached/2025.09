# Fine-Grained Visual Identification Engine

Hand-written CUDA inference engine, demonstrated on fine-grained vehicle identification.

---

## 0. Read this first

**This is a GPU performance engineering project that happens to run an ML workload.** It is not an ML project. The resume value is in the CUDA kernels, the hand-written inference executor, and the measured performance work. The car identification is a harness that makes the demo legible and gives the kernels something real to run.

When any decision is ambiguous, resolve it in favour of *more hand-written GPU code and better measurement*, not in favour of better model accuracy or more product features.

The author is a C++ systems engineer (~1 year production C++ on Linux: database proxy layer, TLS/mTLS, access-control enforcement). Strong on C++, systems, SQL, TDD. **Deliberately light on ML theory.** Explain ML concepts in plain language when they come up; do not assume familiarity with ML jargon. Do assume full competence in C++, build systems, debugging, and server architecture.

---

## 1. What it does

Upload a photo of a car → get back the exact make/model/year, plus visually similar cars from a catalog. Filterable by make, body type, year.

Architecture is **retrieval, not classification**:

```
JPEG on disk
   ↓  [OUR CUDA KERNELS] decode, resize, normalise, layout transform
tensor [N, 3, 224, 224]
   ↓  [OUR C++ GRAPH EXECUTOR] walks the ONNX graph, dispatches each op
   ↓  [OUR CUDA KERNELS] matmul, softmax, layernorm, GELU, ...
   ↓  using pretrained + fine-tuned weights loaded from the .onnx file
512-dim embedding ("fingerprint")
   ↓
nearest-neighbour search over catalog embeddings (pgvector / GPU brute force)
   ↓
identification + similar vehicles
```

**Why retrieval over classification:** new vehicle models are added by inserting embeddings, with no retraining. A fixed 196-way classification head must be retrained to learn a 197th class. This is a deliberate architectural choice and a key talking point.

---

## 2. Hardware and environment

| | |
|---|---|
| GPU | NVIDIA RTX 4060, 8 GB GDDR6 |
| Architecture | Ada Lovelace, compute capability **sm_89** |
| Memory bandwidth | ~272 GB/s theoretical (128-bit bus) — **this is the binding constraint** |
| FP32 compute | ~15 TFLOPS |
| Tensor cores | 4th gen (FP16, BF16, INT8, FP8 available) |
| OS / toolchain | **Windows + Visual Studio + MSVC**, CMake |
| Database | PostgreSQL 16, local install (not Docker) |

**Platform note:** Windows/MSVC is the working environment and is a fully supported CUDA configuration. Do not suggest migrating to WSL2 or Linux. Keep the core engine portable (standard C++, CMake, no Win32 in the engine layer) so that Linux/Docker remains a cheap option at week 11 if the deployment story calls for it. Confine any platform-specific code to a thin isolated layer.

**The 4060 is compute-rich relative to its bandwidth.** Memory-bound kernels are severely memory-bound on this card. This is pedagogically useful: coalescing, tiling, and fusion produce large, obvious speedups. Roofline analysis will be clearly legible. Lean into memory-bound optimisation.

---

## 3. Hand-written vs off-the-shelf

This distinction is the entire point of the project. Get it wrong and the project has no value.

### Hand-written (ours)
- All CUDA kernels: preprocessing, matmul, softmax, layernorm, GELU, elementwise ops
- The **ONNX graph executor** — parses the model file, walks the op graph, dispatches to our kernels, manages device memory
- The C++ serving layer: request handling, dynamic batching, backpressure, lifecycle
- The benchmark harness
- Vector search (GPU brute force first, then indexed)

### Off-the-shelf (allowed)
- **Pretrained weights** — CLIP ViT-B/32. Nobody expects us to have trained 88M parameters.
- **ONNX Runtime — AS A CORRECTNESS ORACLE ONLY.** Never in the serving path.
- **cuBLAS — AS A BENCHMARK OPPONENT ONLY.** Never in the serving path.
- **PyTorch — for the offline fine-tune only** (week 9), then export to ONNX. Training is offline batch work; it does not need to be C++.
- Protobuf (ONNX parsing), libjpeg/nvJPEG (decode), Postgres client lib, a test framework.

### Hard rule
> If ONNX Runtime, cuBLAS, or TensorRT appears anywhere in the inference path that serves a user request, the project has failed its purpose.

They may be linked into separate benchmark and test binaries. They must not be linked into the serving binary.

---

## 4. Non-negotiable principles

1. **Measurement infrastructure before optimisation.** The benchmark harness is built in week 1, before any application kernel. Every performance claim flows through it.

2. **`BENCHMARKS.md` is append-only.** Every measurement gets a dated entry: kernel version, configuration, hardware state, the number, and what changed. Never overwrite, never delete. This catches regressions and it means the week-12 write-up is already largely written.

3. **Correctness is verified against an oracle, always.** Two independent oracles exist:
   - *Kernel level:* our output vs ONNX Runtime's output on identical input, within float tolerance.
   - *Model level:* our fine-tuned accuracy vs published Stanford Cars benchmark numbers.
   A kernel that is fast and wrong is worth nothing. Wire up the oracle comparison before optimising anything.

4. **Profile before optimising, and explain why.** Nsight Compute output backs every optimisation. "I made it faster" is worth far less than "it was memory-bound at X% of peak bandwidth because of uncoalesced access; here is the fix and the new number."

5. **The domain is configuration, never code.** No string `"car"` anywhere in the engine. Dataset path, class count, label schema, filter facets, embedding collection — all in a YAML config. Swapping to flowers or birds must be a config edit plus a training run. This is cheap now and expensive to retrofit.

6. **Daily commits.** The commit history is part of the artifact. It is what converts an employment gap into visible sustained work.

7. **Benchmarks are run with locked GPU clocks**, after warmup iterations, timed with CUDA events (not wall clock), reporting median and p99 over N repeats — never a single number.

---

## 5. Repository layout

```
/
├── CLAUDE.md                 # this file
├── README.md                 # the public-facing artifact — treat as a deliverable
├── BENCHMARKS.md             # append-only measurement log
├── CMakeLists.txt
├── configs/
│   └── cars.yaml             # domain config: dataset, classes, facets
├── src/
│   ├── bench/                # harness: clock lock, warmup, CUDA event timing, stats
│   ├── kernels/              # all hand-written CUDA
│   │   ├── preprocess/       # resize, normalise, NHWC→NCHW
│   │   ├── gemm/             # matmul, versioned v1..vN — KEEP ALL VERSIONS
│   │   └── nn/               # softmax, layernorm, gelu, elementwise
│   ├── engine/               # ONNX parsing, graph walk, dispatch, memory management
│   ├── search/               # embedding storage + nearest neighbour
│   └── server/               # C++ service, batching, backpressure
├── tests/                    # correctness vs ONNX Runtime oracle
├── tools/                    # benchmark binaries, profiling scripts
└── python/                   # offline only: fine-tune, ONNX export, catalog embedding
```

**Keep every matmul version.** `gemm_v1_naive`, `gemm_v2_coalesced`, `gemm_v3_tiled`, and so on, all compiled and all benchmarkable side by side. The progression *is* the deliverable — deleting the slow versions destroys the story.

---

## 6. Model choice

**CLIP ViT-B/32 vision encoder.** Fixed. Reasons:

- It is transformer-based, so the operations are overwhelmingly matrix multiplies. The optimised matmul from weeks 2–4 becomes the actual engine the network runs on — not a side exercise. Every percent gained shows up in end-to-end latency.
- A convolutional backbone (ResNet) would make convolution the headline op, which is harder to implement from scratch and would decouple the matmul work from the product.
- It outputs a 512-dim embedding natively — exactly the fingerprint we want.
- Input: `[N, 3, 224, 224]` float32.

**Op set required** — roughly 10–14 ops, not the ~200 in the ONNX spec:
`Conv` (patch embedding — stride equals kernel size, so implementable as a matmul), `MatMul`/`Gemm`, `Add`, `Mul`, `LayerNormalization`, `Softmax`, `Gelu`/`Erf` (CLIP uses QuickGELU — verify against the exported graph), `Reshape`, `Transpose`, `Concat` (CLS token), `Slice`, `Gather`.

**Do not build a general ONNX runtime.** Support exactly the ops this one model needs, fail loudly on anything else, and say so explicitly in the README. That is a legitimate and defensible scope.

**Preprocessing convention** — CLIP-specific, and silent failure if wrong (shapes still align, output is just quietly garbage):
- Resize to 224×224 (bicubic, shorter side then centre crop)
- Scale 0–255 → 0.0–1.0
- Normalise per channel: mean ≈ `(0.4815, 0.4578, 0.4082)`, std ≈ `(0.2686, 0.2613, 0.2758)`
- RGB channel order, NCHW layout

**Verify these values against the actual model card before trusting them**, and verify the whole preprocessing path against a Python reference on day one of week 5.

---

## 7. Twelve-week plan

| Week | Focus | Definition of done |
|---|---|---|
| **1** | Setup, benchmark harness, bandwidth baseline | Repo live, CMake builds, harness works, vector-add achieved bandwidth recorded vs 272 GB/s theoretical |
| **2–4** | **Matmul.** naive → coalesced → tiled → register-blocked → vectorised | Each version benchmarked as % of cuBLAS, with Nsight analysis explaining each jump. Target >50% of cuBLAS by end of week 4 |
| **5** | Preprocessing kernels | JPEG → correct tensor, output matches Python reference within tolerance |
| **6–7** | ONNX parsing + graph executor | CLIP ViT-B/32 runs end to end on our kernels |
| **8** | Remaining kernels + correctness | Our embeddings match ONNX Runtime within float tolerance, all ops |
| **9** | Fine-tune on Stanford Cars (Python), export, embed catalog | Accuracy near published benchmarks; catalog embedded in Postgres |
| **10** | Vector search: GPU brute force → indexed | Recall vs latency curve across index configurations |
| **11** | C++ serving layer: batching, backpressure, load test | p50/p95/p99 under sustained load; does not fall over or OOM |
| **12** | Second domain, docs, write-up | Two domains working via config swap; README complete |

**Weeks 2–4 are the heart of the project. Do not compress them.** That block produces the single sentence a hiring manager stops scrolling for.

**Checkpoints:**
- End of week 4 below ~50% of cuBLAS → stop and debug the approach rather than pushing forward.
- Week 7 slipping → cut the week-12 second domain first. It is the most expendable piece.

---

## 8. Week 1 in detail — start here

1. Confirm the CUDA toolchain builds from Visual Studio/CMake for `sm_89`.
2. Create the GitHub repo **today**, even empty. README with one paragraph on what this is.
3. Build the benchmark harness:
   - lock GPU clocks (`nvidia-smi` from an elevated prompt) and record the state
   - warmup iterations before timing
   - time with **CUDA events**, not wall clock
   - N repeats, report **median and p99**, plus the spread
4. Write `vector_add` as the first kernel. It does almost no arithmetic, so it is purely bandwidth-limited.
5. Compute achieved bandwidth: bytes moved ÷ time. Compare against ~272 GB/s. Expect 70–85%.
6. **Run it 20 times and check the variance.** Under a few percent → the measurement environment is trustworthy. Swinging 15–20% → investigate thermal throttling or WDDM scheduling noise before proceeding. Every number for the next three months rests on this.
7. Record it all in `BENCHMARKS.md` with methodology.

That achieved-bandwidth figure is the project's speed limit. For the remaining eleven weeks the recurring question about every kernel is: *how close to that ceiling, and if far, why?*

---

## 9. Domain portability

The GPU stack is domain-agnostic — it sees a `[N, 3, 224, 224]` float array and nothing more. Swapping domains changes only the dataset, the fine-tuned weights, and the catalog metadata.

**Week 12 second domain:** Oxford 102 Flowers or CUB-200 Birds. Both are clean standard benchmarks with published accuracy numbers, giving a second correctness oracle. Budget 3–4 days.

Avoid domains with no standard dataset (they turn into a week of hand-labelling) and avoid datasets shot on plain white backgrounds (accuracy saturates near 99%, nothing interesting to discuss).

**Keep embedding collections separated per domain** — a rose must not compete with a Civic in the same nearest-neighbour search.

Two domains proves generality. Three is padding.

---

## 10. Do not

- Do not use ONNX Runtime, cuBLAS, or TensorRT in the serving path.
- Do not write a general-purpose ONNX runtime. Support this model's ops, fail loudly otherwise.
- Do not write a training loop in CUDA. Fine-tuning is offline Python.
- Do not hardcode `"car"` anywhere in the engine.
- Do not delete old kernel versions. The progression is the deliverable.
- Do not optimise before the correctness oracle is wired up.
- Do not claim a speedup without a `BENCHMARKS.md` entry and a profiler trace behind it.
- Do not suggest migrating to WSL2 or Linux. Windows/MSVC is the environment.
- Do not add product features (video, mobile, auth, multi-user) before week 11. Feature creep is how a 12-week project becomes a 16-week unfinished one.
- Do not store training images in Postgres. Training data lives on disk; Postgres holds the *catalog* — metadata and embeddings.

---

## 11. Reading

- **siboehm, "How to Optimize a CUDA Matrix Multiplication Kernel"** — naive to near-cuBLAS across ten kernels with measurements at each step. This is essentially weeks 2–4. Read first, then implement independently without copying.
- **Kirk & Hwu, *Programming Massively Parallel Processors*** — chapters 1–6.
- **NVIDIA CUDA C++ Best Practices Guide** — reference.
- **Nsight Compute** — learn it in week 2, not week 8. Reading a profiler is the transferable skill.

---

## 12. Framing

In the README and on the resume, this is:

> **A from-scratch CUDA inference engine, demonstrated on fine-grained vehicle identification.**

Not "a car identification system." The first tells a reader what was actually built.

Resume bullets should carry numbers: *"custom CUDA GEMM at X% of cuBLAS on RTX 4060"*, *"hand-written ONNX graph executor, N images/sec at batch 32, p99 Y ms"*, *"vector search over N embeddings at Z% recall."*
