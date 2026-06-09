# DeepSeek V4 CPU Optimization TODO

This branch is now in optimization mode. Keep FA-on as the primary benchmark
path; use FA-off as a correctness/stability regression gate.

## Immediate Profiling

- [x] Profile Q4 FA-on, single node-local private copy:
  `numactl --physcpubind=0-25 --membind=0 ... --no-mmap -t 26 -tb 26`.
- [x] Profile Q4 FA-on, all physical cores with interleaved private copy:
  `numactl --physcpubind=0-51 --interleave=all ... --no-mmap -t 52 -tb 52`.
- [x] Compare the profile against the earlier mmap/cross-socket baseline.
- [x] Separate prefill and decode-heavy costs. The current benchmark is
  decode-dominated; a prefill-heavy prompt may prefer a different policy.
- [x] Use `DSV4_DEBUG_SECTION_TIMING=1` and `DSV4_DEBUG_DECODE_TIMING=1`
  first, then use `perf` only for the hottest confirmed buckets.

### Profiling Snapshot 2026-06-09

Commands used the Q4_K_M-XL GGUF, FA-on, `--no-mmap`, ctx 1024, fixed
620-token prompt, and `n_predict=96`. The section-timing callback synchronizes
at graph markers, so use proportions rather than absolute latency.

Node-local private copy, `numactl --physcpubind=0-25 --membind=0 -t 26 -tb 26`:

```text
profile dir:
  dsv4-cascade-lake-results/profile-node0-no-mmap-fa-on

one-token decode timing:
  build   mean 4.58 ms
  alloc   mean 8.71 ms
  inputs  mean 0.07 ms
  compute mean 323.61 ms

top section buckets:
  ffn_moe_out              13.24 s
  attn_out                  9.32 s
  Qcur                      7.71 s
  dsv4_compressed_attn_out  5.20 s
  hc_attn_post              3.73 s
  KVcur                     3.70 s
  hc_ffn_post               3.60 s
  hc_attn_pre               3.18 s
```

All physical cores with interleaved private copy,
`numactl --physcpubind=0-51 --interleave=all -t 52 -tb 52`:

```text
profile dir:
  dsv4-cascade-lake-results/profile-allphys-no-mmap-interleave-fa-on

one-token decode timing:
  build   mean 5.01 ms
  alloc   mean 8.83 ms
  inputs  mean 0.08 ms
  compute mean 434.67 ms

top section buckets:
  ffn_moe_out              11.06 s
  Qcur                     10.36 s
  attn_out                 10.22 s
  hc_attn_post              7.16 s
  hc_ffn_post               7.02 s
  dsv4_compressed_attn_out  5.51 s
  KVcur                     3.77 s
  hc_attn_pre               3.47 s
```

Interpretation:

- Decode is compute/memory dominated; graph build is not the first target.
- All physical cores improve the MoE bucket but make Q projection and HC post
  helpers worse, consistent with remote-memory penalties from a split private
  model copy.
- The next narrow code-level work should start with MoE/output projection and
  HC helper fusion or locality, not `DSV4_HC_SPLIT_SINKHORN`.
- Userspace `perf` profiling is available now. For the node-local no-mmap
  request, `cycles:u` samples show:

```text
profile dir:
  dsv4-cascade-lake-results/perf-node0-no-mmap-fa-on

top self samples:
  gomp_team_barrier_wait_end                 23.88%
  Q8_0_1_Unpacker mul_mat helper             16.02%
  ggml_barrier                               11.29%
  Q4K AVX2 selected-expert/matmul helper      9.05%
  q8_0_r8_q8_2 helper                         4.86%
  Q6K AVX2 selected-expert helper             2.66%
  q8_1_r8_q8_2 helper                         2.38%
  ggml_compute_forward_flash_attn_ext_f16     2.01%
  ggml_compute_forward_mul_f32                1.32%
  ggml_compute_forward_concat                 1.29%

custom DSV4 ops:
  ggml_compute_forward_dsv4_fp8_kv_quantize   0.72%
  ggml_compute_forward_dsv4_hc_split_sinkhorn 0.02%
```

- The two retained custom DSV4 ops are not first-order optimization targets for
  this prompt. `FP8_KV_QUANTIZE` can wait unless a longer-context profile shows
  it growing. `HC_SPLIT_SINKHORN` should stay scalar/reference for now.
- Synchronization/barrier overhead is large in the perf sample. Some of that is
  normal OpenMP scheduling around many graph nodes, but it strengthens the case
  for reducing small HC/helper nodes or improving graph fusion.

## Low-Hanging Runtime Sweeps

- [x] Add a first-class `--no-mmap` option to `scripts/engine_test_harness.py`
  or wrap the server command in a reusable script so no-mmap runs are not
  one-off Python snippets.
- [x] Repeat best no-mmap NUMA policies:
  - single node0 private copy, `26/26`;
  - single node1 private copy, `26/26`;
  - all physical cores, interleaved private copy, `52/52`;
  - two node-local private-copy servers for aggregate serving.
- [x] Re-run the best policies on the BLIS build. BLIS improved the server
  harness before, but the no-mmap/NUMA interaction has not been measured.
- [x] Run a longer decode-heavy prompt and a larger prefill-heavy prompt so the
  recommendation is not overfit to the current 620-token prompt.
- [x] Install or expose `perf` and run userspace symbol profiles.
- [ ] Re-run `perf` on a longer context/decode to see whether FP8 KV quantize
  grows with cache length.
- [ ] Use perf call stacks plus section markers to separate:
  - routed MoE `mul_mat_id` kernels;
  - ordinary projection matmuls;
  - HC primitive `mul/sum_rows/concat` overhead;
  - OpenMP barrier time caused by many small graph nodes.

## Promising Larger Patches

- [ ] Fix DSV4 `-np > 1` / multi-slot server batching. It currently aborts in
  `build_deepseek4`, so one-server continuous batching is not a valid baseline.
- [ ] Prototype NUMA-replicated read-only model weights. A real implementation
  would let one process use all cores while each socket reads local copies.
  Treat this as a backend/runtime project, not a small graph patch.
- [ ] Investigate HC helper fusion. HC post helpers have shown up as a hot
  bucket, and the current primitive decomposition creates many graph nodes.
- [ ] Revisit the `ggml_mul_mat(hc_fn, flat)` HC-pre mismatch. Recovering the
  optimized primitive path would be cleaner than carrying scalar-equivalent row
  decomposition forever.
- [ ] Deprioritize `DSV4_FP8_KV_QUANTIZE` for the current short-context FA-on
  path. It was only 0.72% self time in the node-local no-mmap perf profile.
  Reconsider for longer contexts.
- [ ] Keep `DSV4_HC_SPLIT_SINKHORN` scalar/reference unless a future profile
  contradicts the current 0.02% self-time result.

### Low-Hanging Sweep Snapshot 2026-06-09

All commands used Q4_K_M-XL, FA-on, the current `build-cpu-opt` binary unless
otherwise noted, and the dual-engine harness with the existing calibrated
logprob gates. `PASS_WITH_WARNING` below means exact generated text/tokens and
hard logprob pass, with small warning-band drift relative to the cached
cross-socket baseline.

```text
node0 no-mmap, 26/26:
  command:
    python3 scripts/engine_test_harness.py --flash-attn ... \
      --ik-server-prefix "numactl --physcpubind=0-25 --membind=0" \
      --ik-numa numactl --ik-threads 26 --ik-threads-batch 26 --ik-no-mmap
  result:
    PASS_WITH_WARNING
    prefill 31.1264 tok/s, decode 3.5608 tok/s, wall 70.47 s

node1 no-mmap, 26/26:
  result:
    PASS_WITH_WARNING
    prefill 30.6682 tok/s, decode 3.4311 tok/s, wall 72.68 s

all physical cores, interleaved no-mmap, 52/52:
  result:
    PASS
    prefill 34.7611 tok/s, decode 3.2777 tok/s, wall 72.75 s

GCC + BLIS, node0 no-mmap, BLIS_NUM_THREADS=1:
  result:
    PASS_WITH_WARNING
    prefill 31.2578 tok/s, decode 3.7541 tok/s, wall 67.78 s
```

The current best low-risk operating policy for single-request latency is still
a private node-local model copy. BLIS with node-local no-mmap is a promising
build candidate, but should remain measured instead of becoming the default
until it repeats under Q8 and a second prompt.

The longer decode-heavy run (`n_predict=384`, fixed 620-token prompt) preserved
the same conclusion:

```text
baseline cross-socket mmap: prefill 33.4853 tok/s, decode 3.0468 tok/s, wall 77.59 s
node0 no-mmap 26/26:       prefill 31.0519 tok/s, decode 3.6561 tok/s, wall 69.20 s
status: PASS_WITH_WARNING, exact text/tokens
```

The 4096-context prefill-heavy probe auto-sized to 3969 prompt tokens, but
`n_predict=64` truncated the strict canary and therefore is not a parity gate.
It is still useful as a perf-only signal:

```text
baseline cross-socket mmap: prefill 28.1710 tok/s, decode 2.2406 tok/s, wall 169.45 s
node0 no-mmap 26/26:       prefill 30.4806 tok/s, decode 2.5912 tok/s, wall 154.91 s
status: incomplete strict report, matching text/tokens for emitted region
```

Do not overfit to the 4096 run until it is repeated with enough generated
tokens for the full canary, but it does not contradict the node-local private
copy direction.

## Guardrails

- [ ] Preserve exact text/token parity and use calibrated logprob thresholds.
- [ ] Keep Q8 forced-F16 KV cache behavior unless explicit quantized-KV gates
  are added and pass.
- [ ] Do not trade FA-on speed for unexplained FA-off NaNs or text divergence.
- [ ] Commit optimization experiments atomically, including rejected results
  when they prevent rediscovery.
