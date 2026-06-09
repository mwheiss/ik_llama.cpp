# DeepSeek V4 CPU Optimization Review

Branch state:

- Baseline branch/build: `deepseek-v4-cpu`, `build-cpu-clx`
- Optimization branch/build: `deepseek-v4-cpu-opt`, `build-cpu-opt`
- Target CPU: Cascade Lake, AVX2/FMA/F16C/AVX-512F/DQ/CD/BW/VL/VNNI, no AMX/BF16/FP16-native/VBMI

## Harness Workflow

`scripts/engine_test_harness.py` now defaults to comparing:

- baseline: `build-cpu-clx/bin/llama-server`
- test: `build-cpu-opt/bin/llama-server`

The baseline run is cached under `.dsv4-baseline-cache/` using a key over the
baseline binary, model, context, flash-attention mode, prompt/request knobs, and
thread count. Use:

```bash
python3 scripts/engine_test_harness.py --refresh-baseline-cache
python3 scripts/engine_test_harness.py
python3 scripts/engine_test_harness.py --flash-attn
```

For quicker opt-loop checks:

```bash
python3 scripts/engine_test_harness.py --ctx-size 1024 --n-predict 192 --filler-lines 0 --refresh-baseline-cache
python3 scripts/engine_test_harness.py --ctx-size 1024 --n-predict 192 --filler-lines 0
```

The second command reuses the cached baseline and runs only `build-cpu-opt`.

The Cascade Lake compiler/BLAS matrix is automated by:

```bash
scripts/build-dsv4-cascade-lake-matrix.sh [case ...]
scripts/bench-dsv4-cascade-lake-matrix.sh [case ...]
```

The build script encodes the safe Cascade Lake target envelope
(`-march=cascadelake`, AVX-512/VNNI on, BF16/VBMI/FP16-native/AMX off), builds
`llama-server`, `llama-bench`, and `test-dsv4-primitives`, runs the primitive
test, verifies VNNI artifacts, and records CMake/ldd/version output under
`dsv4-cascade-lake-results/`. The benchmark script compares each candidate
against `build-cpu-opt` through the deterministic engine harness. FA-on is the
primary/default benchmark mode (`DSV4_FLASH_MODES=on,off`) because it is the
expected performance path for DSV4; FA-off remains in the default sequence as a
numerics and stability regression gate. Optional `llama-bench` PP/TG shapes via
`DSV4_RUN_LLAMA_BENCH=1`.

Verified local toolchain/library stack after installation:

```text
GCC 14.3.1
Intel oneAPI DPC++/C++ Compiler 2026.0.0: icx/icpx
Clang 21.1.8 with libomp
oneMKL 2026.0, MKLROOT=/opt/intel/oneapi/mkl/2026.0
OpenBLAS 0.3.29 via FlexiBLAS OPENBLAS-OPENMP
BLIS 2.0 OpenMP
numactl, ninja, objdump
```

## Current Baseline

Short cached-baseline smoke, flash attention off:

```text
python3 scripts/engine_test_harness.py --ctx-size 1024 --n-predict 192 --filler-lines 0

comparison_status = PASS
max_abs_logprob_diff = 0
mean_abs_logprob_diff = 0
baseline_ik decode = 1.8400 tok/s
opt_ik decode      = 1.8396 tok/s
```

The two builds are still source-equivalent, so exact parity is expected.

## Numerical Reference Policy

The optimization harness can now use a single-threaded baseline as the
numerical anchor and compare parallel/runtime policies against it. This is a
better reference than treating an arbitrary 52-thread run as "truth", because
parallel reductions and scheduling can legitimately move logprobs by small
amounts while leaving token/text output unchanged.

Reference-binary calibration, Q4_K_M-XL, FA-off, ctx=1024, n_predict=192,
filler_lines=0:

```text
case                         max_abs_logprob_diff  mean_abs_logprob_diff  decode tok/s
ref_t52_tb52                 0.0064106201          0.0001467117           1.8784
ref_t32_tb52                 0.0064106201          0.0001467117           2.5107
ref_t104_tb104               0.0094827684          0.0001349938           1.2833
ref_numa_distribute_t32_tb52 0.0064106201          0.0001467117           2.4998
ref_numa_distribute_t52_tb52 0.0064106201          0.0001467117           2.3547
ref_numa0_phys_t26_tb26      0.0131351781          0.0001699866           2.0916
```

Based on that reference-only envelope, FA-off hard failure is now reserved for
`max_abs_logprob_diff > 0.02` or mean drift beyond `5e-4`, while warnings start
above `0.007`. Text/token equality remains strict. The warning band intentionally
marks full-SMT and single-socket-local schedules for review without confusing
them with correctness failures.

Calibration command:

```bash
scripts/calibrate-dsv4-reference-drift.sh build-cpu-clx 1024 192 0 off
```

The same script accepts `on` as the fifth argument for FA-on calibration.

## External And Upstream Clues

- ik upstream documents that quantized GEMM hot paths live in
  `ggml/src/iqk/iqk_gemm_*.cpp`, and that AVX-512 quantized matmul requires
  `AVX512F/VNNI/VL/BW/DQ`; without those macros the build silently falls back to
  AVX2. The docs also state that VNNI is responsible for most quantized-matmul
  speedup. Source: https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md
- ik parameter docs keep fused MoE, fused up/gate, and fused mul-multiadd
  enabled by default, and recommend runtime repack where interleaved variants
  are available. Source: https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md
- ik token-generation performance notes emphasize that too many generation
  threads can oversaturate CPU execution, so thread-count sweeps are a legitimate
  optimization axis even when graph math is unchanged. Source:
  https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/development/token_generation_performance_tips.md
- TeamBlobFish recommends `--no-repack` for V4 GGUFs in cchuter's runtime path
  because repack can materially increase load-time memory pressure. That is a
  load/memory clue, not yet a proven steady-state speed clue for ik. Source:
  https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF
- Fetched upstream ik history highlights recent CPU/MoE/attention work worth
  mining:
  - `6d4cdef5` optimizes `mul_mat_q8_1_r8_q8_2` with AVX-512 for Q4_K/Q5_K
    prompt processing.
  - `55d3c05b` adds a fused RMS-norm + RMS-norm + add op.
  - `4fbd0c44` changes CPU Flash Attention mask handling.
  - `0b81212d` broadens CPU FlashMLA quant coverage.
  - `dd67a9fb` optimizes MLA tensor-parallel prompt processing.

## Ranked Optimization Candidates

### 1. Profile DSV4 graph hot nodes first

Before changing graph math, add low-overhead timing or use existing server/graph
timing to split DSV4 runtime into:

- local/raw attention,
- compressed attention,
- compressor/indexer projections,
- MoE routed up/gate/down,
- hyperconnection helpers,
- final logits/output.

Expected payoff: high. It tells us whether to chase attention, MoE, HC, or
loader/repack first.

Risk: low if instrumentation is runtime-gated and off by default.

Implemented debug tooling:

```bash
DSV4_DEBUG_DECODE_TIMING=1
DSV4_DEBUG_SECTION_TIMING=1
```

For Q4_K_M-XL, FA-off, ctx=1024, n_predict=192,
`--numa distribute -t 32 -tb 52`, true one-token decode is compute dominated:

```text
decode n=1 mean build  =  6.96 ms
decode n=1 mean alloc  = 12.93 ms
decode n=1 mean inputs =  0.08 ms
decode n=1 mean compute=368.41 ms
```

The section-timing callback synchronizes at markers and therefore slows the
run, but its proportions point to these large buckets:

```text
Qcur/query projection                 ~23.6 s total in debug run
MoE routed output                     ~20.4 s
attention output projection/update    ~17.6 s
compressed attention output           ~15.7 s
HC post helpers                       ~22.2 s combined
```

This says graph construction is not the main optimization target for decode;
the next useful work should attack compute-heavy projection, attention, MoE,
or HC helper paths.

### 2. Recover F16/IQK V*KQ for FA-off compressed attention

`NUMERICS_FINDINGS.md` says FA-off currently materializes V as F32 before
`V * softmax(KQ)` because the F16/IQK path produced NaNs on long-cache decode.
This is correctness-first but likely costs memory traffic and decode speed.

Next experiment:

- isolate the DSV4 `dsv4_attn_kqv` shape into a focused numerical test,
- compare generic GGML, IQK-enabled F16, and F32-materialized paths,
- only remove or narrow the F32 cast if the focused repro is finite and the
  cached-baseline harness stays on parity.

Expected payoff: high for FA-off decode.

Risk: high. Previous broad attempts diverged or slowed down.

Tried and rejected so far: replacing the one-query per-head KQV decomposition
with direct `ggml_mul_mat(v_attn, kq)` behind
`DSV4_EXPERIMENT_DIRECT_ONE_QUERY_KQV=1`. It failed after producing only `AL`
and slowed decode to `0.364 tok/s`. Keep the decomposition until the underlying
4D F32/F32 matmul path is debugged with a focused regression test.

Status: not currently actionable as a graph-level optimization. Multiple retry
experiments in `NUMERICS_FINDINGS.md` show that cchuter-style F16 V input
causes immediate text divergence or NaNs in ik's FA-off DSV4 path, including
after the HC weighted-sum layout fix. Narrowly bypassing the IQK KQV path made
the long-cache repro finite but was slower and did not recover parity. Keep the
F32 V materialization guard until the underlying F16/IQK `V * softmax(KQ)`
kernel issue has a focused low-level fix and regression test.

### 3. Check whether upstream AVX-512 Q4/Q5 IQK GEMM improvements are already present

Upstream commit `6d4cdef5` targets Q4_K/Q5_K prompt processing. Q4_K_M-XL uses
Q4_K/Q6_K tensors, so this can matter for prefill and some projections.

Next experiment:

- diff `ggml/src/iqk/iqk_gemm_legacy_quants.cpp` and
  `ggml/src/iqk/iqk_mul_mat.cpp` against upstream,
- if missing, port only that upstream kernel change,
- validate with `test-dsv4-primitives`, VNNI check, and cached FA-off/FA-on
  harness.

Expected payoff: medium-to-high for Q4 prefill.

Risk: medium, but isolated to upstream-proven kernels.

Status: already present. `6d4cdef5` is an ancestor of `deepseek-v4-cpu-opt`,
and the local tree contains the `HAVE_FANCY_SIMD` AVX-512 implementation of
`mul_mat_q8_1_r8_q8_2` plus the `GGML_TYPE_Q8_1` row-count change.

### 4. Look for DSV4 HC helper fusion opportunities

Current primitive HC helpers use repeat/mul/sum/view patterns and special F32
row decomposition for HC-pre. They are numerically validated, but create many
small graph nodes. Candidate fusions:

- a DSV4-only fused HC weighted sum CPU kernel,
- a safe optimized `ggml_mul_mat(hc_fn, flat)` replacement once the layout issue
  in `NUMERICS_FINDINGS.md` is isolated,
- reuse upstream fused norm/add style if HC update patterns map cleanly.

Expected payoff: medium, especially decode where node overhead matters.

Risk: medium. HC was a major parity trap; keep focused numerical tests before
the full harness.

### 5. Revisit DSV4 shared expert and MoE path

The current DSV4 shared expert intentionally uses a cchuter-style explicit path
because it improved local parity. The routed path already keeps ik fused MoE and
`ggml_multi_add` enabled. Possible next steps:

- profile whether shared expert or routed expert dominates decode,
- inspect upstream MoE small-batch work even though commit `2973e809` is CUDA
  only; the scheduling idea may still apply to CPU selected-expert work,
- check whether any upstream CPU MoE/fused-mmad changes after our base can be
  cherry-picked without DSV4-specific divergence.

Expected payoff: medium.

Risk: medium. Previous ordered-reduction and disabled-fusion experiments did
not improve parity.

Latest source-level retry: replacing the explicit DSV4 shared expert expression
with the generic ik `llm_build_ffn(..., LLM_FFN_PAR)` helper preserved exact
FA-on logprob parity but did not improve the primary server-generation metric:

```text
command:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 \
    scripts/bench-dsv4-cascade-lake-matrix.sh gcc-native

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline:    prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
generic FFN shared exp: prefill 33.3067 tok/s, decode 2.5959 tok/s, wall 87.9563 s
comparison: PASS with exact logprobs
```

This is a small prefill-only gain with a decode/total-wall regression, so keep
the explicit cchuter-style DSV4 shared expert path. Revisit only if a later
profile shows a different shared-expert bottleneck or if upstream changes the
generic FFN helper enough to require a fresh measurement.

### 6. Thread and batch sweep on Cascade Lake

The harness currently uses full `-t 52 -tb 52`. Upstream docs warn that too many
threads can oversaturate token generation. Run cached-baseline sweeps for opt:

```text
-t/-tb: 16, 24, 32, 40, 52
flash-attn: off/on
ctx: 1024 first, then long prompt
```

Expected payoff: medium and quick, especially if decode is memory-bound.

Risk: low. This can first be harness-only before any code change.

Initial measured result: `--numa distribute -t 32 -tb 52` is the best short
Q4_K_M-XL policy so far, with exact text/token parity and reference-envelope
logprob behavior. Full SMT oversubscription (`-t 104 -tb 104`) is much slower
and drifts more. Binding the run to node 0 physical cores plus local memory is
also slower and drifts more, so the useful NUMA direction is likely balanced
page/thread placement across sockets rather than forcing this model onto one
socket.

Follow-up FA-on sweep, using the current optimized binary against itself and
the cached canonical baseline (`--numa distribute -t 32 -tb 52`), shows that
NUMA/thread placement is workload-sensitive and should be measured with the
serving shape:

```text
command:
  DSV4_SWEEP_FLASH_MODES=on scripts/sweep-dsv4-cpu-opt.sh 1024 192 0

cached baseline distribute -t 32 -tb 52:
  prefill 32.96 tok/s, decode 2.63 tok/s, wall 87.26 s

same-binary candidates:
  control distribute -t 32 -tb 52:
    exact, prefill 33.10 tok/s, decode 2.59 tok/s, wall 88.15 s
  distribute -t 52 -tb 52:
    exact, prefill 33.09 tok/s, decode 2.70 tok/s, wall 85.44 s
  distribute -t 104 -tb 104:
    warning drift, prefill 28.62 tok/s, decode 2.14 tok/s, wall 105.65 s
  isolate -t 52 -tb 52:
    exact, prefill 28.73 tok/s, decode 3.07 tok/s, wall 80.13 s
  numactl --interleave=all -t 52 -tb 52:
    exact, prefill 33.17 tok/s, decode 2.71 tok/s, wall 84.99 s
  numactl --physcpubind=0-51 --interleave=all -t 52 -tb 52:
    exact, prefill 34.01 tok/s, decode 2.77 tok/s, wall 83.32 s
  numactl --physcpubind=0-25 --membind=0 -t 26 -tb 26:
    warning drift, prefill 28.20 tok/s, decode 2.29 tok/s, wall 100.75 s
  numactl --cpunodebind=0 --membind=0 -t 52 -tb 52:
    exact, prefill 26.94 tok/s, decode 2.17 tok/s, wall 105.80 s
```

The single best `--numa isolate` result did not repeat on an immediate follow-up
run (`wall 105.24 s`, exact logits), so do not switch the single-server default
to `--numa isolate` without more repetitions. The best stable-looking
single-server placement from this sweep is physical cores across both sockets
with interleaved memory:

```text
numactl --physcpubind=0-51 --interleave=all ... -t 52 -tb 52
```

The earlier one-node numbers above used mmap and therefore were not a clean
locality test. A single private node-local run with `--no-mmap` is the fastest
per-request topology measured so far:

```text
command:
  numactl --physcpubind=0-25 --membind=0 \
    build-cpu-opt/bin/llama-server ... --numa numactl --no-mmap -t 26 -tb 26

result:
  prefill 31.59 tok/s, decode 3.78 tok/s, wall 67.21 s
  peak RSS: about 167.5 GiB
```

`numastat -p` confirmed that the single private model copy was almost entirely
on node 0. This explains the apparent contradiction: a single node-local
instance with mmap was slower, but a single node-local instance with a private
node-local model copy is faster than the cached cross-socket baseline for this
decode-heavy request.

Two-instance load balancing is still promising for aggregate throughput. A
dedicated smoke now launches one server per socket and fires two requests
concurrently:

```text
command:
  scripts/bench-dsv4-two-node-servers.py \
    --no-mmap \
    --out dsv4-cascade-lake-results/two-node-no-mmap-fa-on \
    --ctx-size 1024 --n-predict 192 --filler-lines 0

result:
  node0 private copy: prefill 30.67 tok/s, decode 3.73 tok/s, wall 68.50 s
  node1 private copy: prefill 30.07 tok/s, decode 3.55 tok/s, wall 71.33 s
  aggregate generated tokens / total wall: 5.05 tok/s
  aggregate decode-only after first token: 7.10 tok/s
  peak RSS: about 167.5 GiB per process
```

This means each private node-local instance is individually faster end-to-end
than the cached single-server baseline for this request, despite slower prefill,
because decode is materially faster. The single-server no-mmap run above is
slightly faster than either instance in the concurrent pair, so the two-instance
setup should be viewed as a throughput/load-balancing topology rather than a
per-request latency improvement. `numastat -p` confirmed that `--no-mmap` placed
the two private model copies almost entirely on their intended sockets. This is
reasonable when memory capacity allows two full model copies and requests can be
load-balanced across instances.

The mmap/shared-page-cache two-instance variant is much more memory-efficient
but was slower. Both requests completed, but `numastat -p` showed that most
model pages for both processes landed on node 1, so the node0 process read
mostly remote memory. Corrected metrics for that run were about `4.22 tok/s`
aggregate decode-only and `3.34 tok/s` generated tokens per total wall. So mmap
can hurt this specific two-instance NUMA-local strategy.

The simpler one-server `-np 2` comparison is not a valid performance baseline
yet: it produced junk for one request and aborted in `build_deepseek4`. Treat
multi-slot concurrent batching as a separate correctness item before comparing
it against two node-local processes.

### 7. Build-flag experiments

Current `build-cpu-opt` keeps the Cascade Lake-safe flags:

```bash
cmake -B build-cpu-opt -S . -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_BF16=OFF \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_SYCL=OFF \
  -DCMAKE_C_FLAGS=-march=cascadelake \
  -DCMAKE_CXX_FLAGS=-march=cascadelake
```

Safe experiments:

- add `-O3` only if it changes generated code and passes parity,
- try `-flto`/thin LTO if build/link time is acceptable,
- compare GCC vs Clang for this DSV4 path.

Avoid:

- `-march=native` if it enables non-Cascade-Lake features accidentally,
- BF16, VBMI/VBMI2, AVX512-FP16, AMX.

Expected payoff: low-to-medium.

Risk: low if guarded by the VNNI artifact script.

Intel compiler/MKL first result: `icx-mkl` builds and passes
`test-dsv4-primitives` plus the Cascade Lake VNNI artifact check, but is not a
good default for the current DSV4 FA-on server path:

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh icx-mkl

benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh icx-mkl

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline: prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
icx + oneMKL:        prefill 32.6452 tok/s, decode 1.5988 tok/s, wall 131.5751 s
comparison: hard pass, warning band; max_abs=0.014243629, mean_abs=0.000231890
```

The text/token output matched, but the decode regression is large enough that
`icx-mkl` should not replace the current GCC native build. Keep testing MKL only
as a prefill-oriented side path or with different thread/runtime settings.

Intel compiler-only result: `icx-native` shows the same decode regression and
same warning-band FA-on logprob drift without MKL, so this is primarily an
Intel compiler/runtime issue for the current DSV4 generation path:

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh icx-native

benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh icx-native

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline: prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
icx native:          prefill 33.0904 tok/s, decode 1.5963 tok/s, wall 131.4941 s
comparison: hard pass, warning band; max_abs=0.014243629, mean_abs=0.000231890
```

Do not switch DSV4 CPU default builds to IntelLLVM unless a later focused
profile identifies and fixes the decode slowdown.

Intel IPO/MKL attempt: `icx-mkl-ipo` configured successfully, but the build was
stopped after the `libggml.so` IPO link remained active for more than 14 minutes
on Cascade Lake:

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh icx-mkl-ipo

observed state:
  /usr/bin/ld ... -plugin /opt/intel/oneapi/compiler/2026.0/.../icx-lto.so ...
  target: ggml/src/libggml.so
  elapsed on same link step: >14 minutes at ~100% CPU
```

Because both `icx-native` and `icx-mkl` already regressed decode throughput, IPO
is not worth more iteration until a specific Intel compiler/codegen issue is
identified. The matrix script now marks interrupted/incomplete builds as failed
instead of allowing stale `PASS` status files.

GCC + MKL result: linking MKL under GCC preserves exact logits, but hurts both
prefill and decode in the FA-on DSV4 server harness:

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh gcc-mkl

benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-mkl

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline: prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
GCC + oneMKL:        prefill 32.4363 tok/s, decode 1.5610 tok/s, wall 134.4268 s
comparison: PASS with exact logprobs
```

This isolates the MKL/BLAS path as harmful for short DSV4 server generation
even without changing compiler. Keep no-BLAS GCC as the default.

GCC + FlexiBLAS/OpenBLAS result: `gcc-flexiblas-openblas` is the first BLAS
candidate that improves the DSV4 server harness while preserving exact logits.
The build links `/usr/lib64/libflexiblas.so` and the benchmark script sets
`FLEXIBLAS=OPENBLAS-OPENMP` with `OPENBLAS_NUM_THREADS=1`.

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh gcc-flexiblas-openblas

FA-on benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-flexiblas-openblas

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline:      prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
GCC + FlexiBLAS/OpenBLAS: prefill 33.3166 tok/s, decode 2.8594 tok/s, wall 81.5600 s
comparison: PASS with exact logprobs

FA-off regression benchmark:
  DSV4_FLASH_MODES=off DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-flexiblas-openblas

FA-off, same prompt/policy
GCC native baseline:      prefill 31.5584 tok/s, decode 2.1935 tok/s, wall 101.7068 s
GCC + FlexiBLAS/OpenBLAS: prefill 31.3067 tok/s, decode 2.4891 tok/s, wall 92.1200 s
comparison: PASS with exact logprobs
```

This is a promising build-path optimization. Before adopting it as the default,
repeat with BLAS thread sweeps and llama-bench/prefill-focused runs to confirm
the win is stable and not only a harness scheduling artifact.

GCC + BLIS result: `gcc-blis` is slightly faster than the FlexiBLAS/OpenBLAS
candidate in this harness, again with exact logits. The build links directly to
`/usr/lib64/libblis.so`.

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh gcc-blis

FA-on benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-blis

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline: prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
GCC + BLIS:          prefill 33.5682 tok/s, decode 2.8750 tok/s, wall 81.0777 s
comparison: PASS with exact logprobs

FA-off regression benchmark:
  DSV4_FLASH_MODES=off DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-blis

FA-off, same prompt/policy
GCC native baseline: prefill 31.5584 tok/s, decode 2.1935 tok/s, wall 101.7068 s
GCC + BLIS:          prefill 31.8438 tok/s, decode 2.5354 tok/s, wall 90.4640 s
comparison: PASS with exact logprobs
```

This is the current best library/build candidate in the matrix. Next steps are
BLIS thread sweeps, a direct comparison against FlexiBLAS/OpenBLAS variability,
and llama-bench/prefill-heavy checks.

BLIS thread sweep result: all FA-on runs preserved exact logits, but additional
BLIS worker threads did not produce a stable improvement over one BLIS thread
inside the already-threaded server. Best repeated/conservative setting remains
`BLIS_NUM_THREADS=1`.

```text
command:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=2,4,8,16,26,52 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-blis
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-blis

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
BLIS threads=1:  prefill 33.1257 tok/s, decode 2.9019 tok/s, wall 80.7439 s
BLIS threads=2:  prefill 33.1566 tok/s, decode 2.8631 tok/s, wall 81.5682 s
BLIS threads=4:  prefill 33.1149 tok/s, decode 2.8677 tok/s, wall 81.4915 s
BLIS threads=8:  prefill 33.1905 tok/s, decode 2.8977 tok/s, wall 80.7981 s
BLIS threads=16: prefill 33.3072 tok/s, decode 2.8775 tok/s, wall 81.1694 s
BLIS threads=26: prefill 33.0133 tok/s, decode 2.8779 tok/s, wall 81.3258 s
BLIS threads=52: prefill 33.1874 tok/s, decode 2.8651 tok/s, wall 81.5064 s
comparison: PASS with exact logprobs for all listed runs
```

This suggests the BLIS win is from the linked BLAS path itself rather than from
oversubscribing BLAS worker threads on top of ik's OpenMP execution.

Quick `llama-bench` check: a single-repetition FA-on mixed shape did not
reproduce the server-harness BLIS win. This makes BLIS a promising
`llama-server` path for the deterministic harness, not yet a universal
throughput win:

```text
command, baseline:
  numactl --interleave=all build-cpu-opt/bin/llama-bench ... -ngl 0 -fa 1 --numa distribute -t 52 -tgb 52 -p 512 -n 128 -r 1

command, BLIS:
  BLIS_NUM_THREADS=1 numactl --interleave=all build-cpu-gcc-blis/bin/llama-bench ... -ngl 0 -fa 1 --numa distribute -t 52 -tgb 52 -p 512 -n 128 -r 1

baseline: pp512 46.09 tok/s, tg128 3.21 tok/s
BLIS:     pp512 45.76 tok/s, tg128 3.19 tok/s
```

Keep the engine harness as the primary optimization gate for generation, and
use llama-bench as a secondary workload-specific sanity check.

FlexiBLAS/OpenBLAS thread sweep result: exact parity also held for all tested
thread counts. Performance is competitive with BLIS but similarly noisy, and the
best single high-thread result did not repeat.

```text
command:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=2,4,8,16,26,52 scripts/bench-dsv4-cascade-lake-matrix.sh gcc-flexiblas-openblas

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
Flexi/OpenBLAS threads=1:  prefill 33.3166 tok/s, decode 2.8594 tok/s, wall 81.5600 s
Flexi/OpenBLAS threads=2:  prefill 33.1422 tok/s, decode 2.8823 tok/s, wall 81.1567 s
Flexi/OpenBLAS threads=4:  prefill 33.1555 tok/s, decode 2.9062 tok/s, wall 80.6358 s
Flexi/OpenBLAS threads=8:  prefill 33.2795 tok/s, decode 2.8888 tok/s, wall 80.9401 s
Flexi/OpenBLAS threads=16: prefill 33.2835 tok/s, decode 2.8889 tok/s, wall 80.9361 s
Flexi/OpenBLAS threads=26: prefill 32.6722 tok/s, decode 2.8799 tok/s, wall 81.4792 s
Flexi/OpenBLAS threads=52: prefill 32.8671 tok/s, decode 2.8786 tok/s, wall 81.3945 s
comparison: PASS with exact logprobs for all listed runs
```

FlexiBLAS/OpenBLAS remains a valid alternative, but the direct BLIS build is a
cleaner recommendation for now because its conservative single-thread setting is
simple and repeatably in the same performance band.

Clang compiler-only result: `clang-native` gives a prefill gain and exact
logprob parity, but decode regresses even more than IntelLLVM:

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh clang-native

benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh clang-native

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline: prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
Clang native:        prefill 34.5014 tok/s, decode 1.5383 tok/s, wall 134.9833 s
comparison: PASS with exact logprobs
```

Clang may be interesting for prefill-heavy microbenchmarks, but it should not
replace GCC for DSV4 generation unless decode is fixed or separately routed.

Clang + MKL result: `clang-mkl` preserves exact logits and keeps the Clang
prefill gain, but decode remains far slower than the GCC native baseline:

```text
build:
  DSV4_CPU_MATRIX_JOBS=104 scripts/build-dsv4-cascade-lake-matrix.sh clang-mkl

benchmark:
  DSV4_FLASH_MODES=on DSV4_BLAS_THREADS=1 scripts/bench-dsv4-cascade-lake-matrix.sh clang-mkl

FA-on, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
GCC native baseline: prefill 32.9625 tok/s, decode 2.6297 tok/s, wall 87.2581 s
Clang + oneMKL:      prefill 34.2386 tok/s, decode 1.5586 tok/s, wall 133.5948 s
comparison: PASS with exact logprobs
```

This confirms that the Clang prefill clue does not become a generation win by
adding MKL. Keep GCC native as the generation default.

Status: `Release` already compiles with `-O3 -DNDEBUG`, so there is no separate
O3-only win to test. A separate GCC LTO build (`GGML_LTO=ON`) passed
`test-dsv4-primitives` and the Cascade Lake VNNI artifact check, but did not
produce a useful speedup:

```text
FA-off, ctx=1024, n_predict=192, --numa distribute -t 32 -tb 52
non-LTO: prefill 31.2958 tok/s, decode 2.4976 tok/s, wall 91.8799 s
LTO:     prefill 31.0562 tok/s, decode 2.5038 tok/s, wall 91.8549 s
logprobs: exact match

FA-on, same prompt/policy
non-LTO: prefill 33.1224 tok/s, decode 2.5611 tok/s, wall 89.0002 s
LTO:     prefill 33.2316 tok/s, decode 2.5584 tok/s, wall 89.0141 s
logprobs: hard pass, warning band; max_abs=0.011420941, mean_abs=0.000164639
```

Because total runtime is effectively flat and FA-on no longer has exact logprob
parity, do not switch the default CPU optimization build to LTO.

Current matrix confirmation (`gcc-lto`) reached the same conclusion: FA-on
passed only with warning-band logprob drift (`max_abs=0.011420941`,
`mean_abs=0.000164639`) and was slower overall than the cached GCC native
baseline (88.1925 s vs 87.2581 s) despite slightly higher prefill throughput.

Clang/ICX comparison is now runnable after installing Clang/libomp and Intel
oneAPI. Use the Cascade Lake matrix scripts above to build and benchmark
`clang-native`, `icx-native`, `icx-mkl`, `gcc-mkl`, `clang-mkl`, and related
BLAS variants before changing the default build recipe.

### 8. Runtime repack and row-interleaved packing

ik docs recommend runtime repack where interleaved variants are available, but
TeamBlobFish warns V4 repack can hurt load memory in cchuter. For ik, measure
steady-state speed and peak RSS separately:

- default repack behavior,
- `--run-time-repack`,
- if supported by this binary, no-repack equivalent or leaving repack off.

Expected payoff: unknown. It may improve GEMM throughput but worsen load-time
memory pressure.

Risk: low if measured in isolation, but do not make it the default until Q4 and
Q8 load behavior is understood.

Status: tested and rejected for the current DSV4 Q4_K_M-XL CPU path. Running
the optimized build with `--run-time-repack` allocated one large CPU buffer of
about 166 GiB and aborted during server initialization in
`iqk_gemm_legacy_quants.cpp` with `GGML_ASSERT(nrc_x%16 == 0) failed`, reached
from `common_speculative_is_compat(llama_context *)`. The optimization harness
now has `--run-time-repack` and `--ik-run-time-repack` switches so this can be
retested later, but runtime repack must not be part of the default DSV4 CPU
policy until the IQK shape constraint and memory footprint are understood.

## Immediate Next Atomic Step

Start with candidate 1: add gated DSV4 graph/tensor timing or use existing
callbacks to produce a hot-node breakdown for the cached-baseline harness. Once
we know the dominant cost, choose one narrow speed change and require:

```bash
git diff --check
cmake --build build-cpu-opt --config Release -j104 --target llama-server test-dsv4-primitives
./build-cpu-opt/bin/test-dsv4-primitives
scripts/check-cascade-lake-vnni.sh build-cpu-opt
python3 scripts/engine_test_harness.py --ctx-size 1024 --n-predict 192 --filler-lines 0
python3 scripts/engine_test_harness.py
python3 scripts/engine_test_harness.py --flash-attn
```

Only commit a speed change if it preserves cached-baseline text/token/logprob
parity and improves at least one relevant throughput metric without a hidden
regression in the other FA mode.
