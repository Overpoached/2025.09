# A from-scratch CUDA inference engine

Hand-written CUDA kernels and a hand-written ONNX graph executor, demonstrated
on fine-grained vehicle identification. No ONNX Runtime, no cuBLAS, no TensorRT
anywhere in the path that serves a request — those are linked into separate
binaries and used only as a correctness oracle and a benchmark opponent.

**Status: week 1 of 12.** The benchmark harness and the bandwidth baseline are
in place. The inference engine is not built yet.

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

Windows, Visual Studio 2026 (toolset v145), CUDA 13.3. Open `cuda.sln` at the
repository root, or from the command line:

```bash
msbuild cuda.sln /p:Configuration=Release /p:Platform=x64
```

Release builds target `sm_89` with `-lineinfo` (for Nsight Compute source
correlation) and without fast-math, so kernel arithmetic stays comparable to the
correctness oracle. **Never benchmark a Debug build** — device code is compiled
with `-G`, and the runner will refuse to call the results trustworthy.

## Running

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
src/bench/              measurement harness: event timing, statistics,
                        NVML clock state, reporting
src/kernels/memory/     vector_add, the bandwidth probe
tools/                  clock locking and profiling scripts
BENCHMARKS.md           append-only measurement log
```

Kernel versions are kept, never replaced. `v1_scalar` and `v2_float4` both
remain compiled and benchmarkable side by side, and the same will hold for the
matmul progression in weeks 2–4. The progression is the deliverable.
