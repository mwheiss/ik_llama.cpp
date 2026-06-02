# DeepSeek V4 Numerics Findings

## HC pre mix projection drift

### Context

DeepSeek V4 hyperconnection pre-mixing computes:

```c++
flat   = rms_norm(reshape_2d(x, n_embd * n_hc, n_tokens))
mixes  = ggml_mul_mat(hc_fn, flat)
split  = ggml_dsv4_hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)
pre    = split[0:n_hc]
post   = split[n_hc:2*n_hc]
comb   = split[2*n_hc:]
y      = hc_weighted_sum(x, pre)
```

cchuter's `v4-clean-base` uses `ggml_mul_mat(hc_fn, flat)` for `mixes`.
In the retry port, the first synthetic HC-pre parity probe exposed a mismatch
at that exact projection step when both inputs were F32.

### Observed discrepancy

The deterministic standalone `mul_mat_f32` probe compared ik against patched
cchuter for a tiny HC-pre-shaped matrix multiply. `flat` was bit-identical
before the projection, but `ggml_mul_mat(hc_fn, flat)` diverged.

Worst observed standalone mismatch:

```text
mul_mat_f32[?]: ik=0.00276000006 cchuter=0.00137999991
```

The expected scalar result is the cchuter value:

```text
expected ~= 0.00137999991
```

The ik result is almost exactly doubled for that element. This is much larger
than ordinary FMA-vs-mul/add float32 rounding. A legitimate rounding-only
difference here should be on the order of a few ULP; the observed error is
about 100% relative error for the worst element.

After replacing the F32 HC mix projection in the staged helper with an explicit
primitive row decomposition:

```c++
row_i = sum_rows(flat * repeat(view(hc_fn[:, i]), flat))
mixes = concat(row_i)
```

the HC-pre parity probe matched cchuter within float32 rounding:

```text
hc_pre_flat    max_abs=0
hc_pre_mixes   max_abs=3.8e-09, worst=1 ULP
hc_pre_pre     max_abs=0
hc_pre_post    max_abs=0
hc_pre_comb    max_abs=0
hc_pre_y       max_abs=3.8e-09, worst=2 ULP
```

The remaining HC weighted-sum helper difference is also small:

```text
hc_weighted_sum max_abs=3e-08, worst=1 ULP
worst: ik=0.319999993 cchuter=0.320000023
```

That magnitude is consistent with operation-order/FMA rounding and is not the
same class of issue as the F32 `ggml_mul_mat` drift above.

### Likely cause

The likely cause is an ik-specific optimized F32 `ggml_mul_mat` path being
selected before the ordinary scalar/F32 fallback. During inspection, current ik
`ggml_compute_forward_mul_mat` can route through IQK optimized matrix
multiplication support for F32-shaped inputs. The cchuter reference does not
appear to use the same path for this case.

Because the worst mismatch was nearly exactly doubled, the issue is unlikely
to be normal FMA contraction or accumulator order. It looks more like a shape,
stride, tiling, row/column interpretation, or fast-path eligibility bug in the
optimized F32 matmul path for this tensor layout.

### Current workaround

For the staged DSV4 HC-pre helper only:

- F32 `hc_fn` uses the explicit scalar-equivalent primitive decomposition.
- Quantized model `hc_fn` still uses normal `ggml_mul_mat`.

This keeps the synthetic parity harness independent of the suspected F32
optimized matmul path while preserving the normal model graph route for GGUF
quantized tensors.

This is intentionally correctness-first and not a final performance strategy.

### What to debug next

To recover the built-in optimized primitive path later:

1. Add a minimal standalone regression test for F32 `ggml_mul_mat` using the
   exact failing HC-pre dimensions, values, strides, and contiguity.
2. Force/disable individual matmul paths and record which path first produces
   the doubled value:
   - scalar fallback
   - BLAS, if enabled
   - IQK optimized matmul
   - any 4D or batched special path
3. Log effective tensor metadata at dispatch:
   - `type`
   - `ne[]`
   - `nb[]`
   - transposition assumptions
   - whether either tensor is contiguous
   - selected kernel/path name
4. Compare the failing output against a simple double-precision reference and
   a strict float32 scalar reference.
5. Try disabling only the suspected F32 IQK fast path for this shape. If that
   restores parity, narrow the fast-path bug before making a broader policy
   change.
6. Once fixed, replace the DSV4-local F32 row decomposition with
   `ggml_mul_mat(hc_fn, flat)` and require the HC-pre parity probe to stay
   within a few ULP.

### Policy until resolved

Do not use the optimized F32 `ggml_mul_mat(hc_fn, flat)` path for HC-pre parity
or scalar reference validation. It can produce errors far beyond expected
rounding. Keep this special handling local to DSV4 HC-pre bring-up unless a
general ik matmul bug is isolated and fixed.

## Grouped output projection recurrence

The same class of issue appeared when adding the DeepSeek4 grouped attention
output helper. A direct F32 implementation using ordinary `ggml_mul_mat` in the
grouped projection produced:

```text
grouped_out[0]: ik=0.00204199972 cchuter=0.000641999999
```

Again, this is much too large to be explained by normal float32 rounding.
After routing the F32 grouped projection substeps through the same
scalar-equivalent row decomposition used for HC-pre validation, the cchuter
parity probe passed:

```text
grouped_out max_abs=7.2e-11, worst=5 ULP
worst: ik=0.000224000061 cchuter=0.000223999989
```

The model graph still uses normal `ggml_mul_mat` for quantized grouped output
weights. The scalar-equivalent projection is currently only a validation path
for F32 helper tests and parity probes.

## DeepSeek4 KV cache type must stay F16 for now

### Context

DeepSeek V4 has more cache state than a standard dense attention model. The
reference implementation maintains:

- local/SWA K/V cache state,
- compressed-attention K/V cache state,
- indexer K cache state.

The compressed-attention graph composes local and compressed K sources. In the
reference path, those K views must agree in dtype because they are concatenated
or otherwise fed through common attention composition primitives.

The model also applies DSV4's own FP8 KV quantize/dequantize behavior before
the ordinary user KV cache type would be applied:

```c++
kv = ggml_dsv4_fp8_kv_quantize(kv, n_rot)
```

So a user-requested `q8_0` KV cache is not simply "store the model's original K
in q8_0"; it is a second, different quantization applied after the DSV4 FP8
activation transform.

### Current policy

For DeepSeek4, force K and V cache types to F16 when the user requests an
unsafe cache type such as:

```text
--cache-type-k q8_0 --cache-type-v q8_0
```

The warning is most visible in Q8 model smokes only because those smokes
intentionally request `q8_0` KV. It is not inherently tied to Q8 model weights.
A Q4 model run with `--cache-type-k q8_0 --cache-type-v q8_0` should trigger
the same override. A default Q4 run usually does not warn because the default
KV cache is already F16.

### Why this is not treated as a harmless speed toggle

Removing the F16 override can fail in two ways:

1. The graph can become invalid if local, compressed, and indexer cache views
   have incompatible dtypes at concat/attention-composition boundaries.
2. Worse, the graph may run but silently corrupt decode because `q8_0`'s block
   scale does not faithfully preserve the post-FP8-quantized K activation
   distribution. The cchuter reference comments report repetitive/gibberish
   decode symptoms from this class of issue.

The second failure mode is especially dangerous for bring-up: it looks like a
model-quality or sampling problem rather than a cache numerics bug.

### Expected speed and accuracy tradeoff

`q8_0` KV could reduce cache memory traffic and footprint for long-context
decode, so it may eventually improve speed or allow larger contexts. It is not
expected to improve accuracy. F16 is the higher-fidelity cache representation
and is the current correctness baseline.

Any future `q8_0` or other quantized DeepSeek4 KV cache support should be
treated as a separate optimization project, not a loader convenience. It needs
explicit gates against the cchuter F16 reference:

1. cache write/read parity for local/SWA, compressed, and indexer caches,
2. dtype compatibility checks for local+compressed K composition,
3. first-token logits or top-k agreement where practical,
4. decode stability on short prompts,
5. sparse compressed-attention context coverage,
6. Q4 and Q8 model-weight smokes with the same forced/unforced cache settings.

Until those gates exist and pass, keep the forced F16 policy.

## Compressed top-k mask scatter is blocked on CPU set_rows

### Context

DeepSeek4 compressed/indexer attention needs a mask with cchuter semantics:

- selected valid compressed rows: `0`,
- unselected compressed rows: `-inf`,
- selected invalid rows: `-1e9`.

cchuter builds this from indexer scores and top-k indices with a scatter-style
primitive:

```c++
mask = ggml_set_rows(mask, values, topk)
```

### Observed issue in ik retry branch

When the retry port added a focused synthetic test for the same helper,
`test-dsv4-primitives` segfaulted inside the CPU implementation:

```text
SIGSEGV in ggml_compute_forward_set_rows
```

There are no other CPU-side users of `ggml_set_rows` in this tree, so this
would introduce an unvalidated primitive into the DSV4 path. The graph might
build, but the helper is not compute-safe on CPU as of this finding.

### Why not replace it with arithmetic masking immediately?

An exact primitive-only replacement is not obvious with the currently used
helpers. Building a mask as arithmetic over `-inf` can produce `NaN` through
terms like:

```text
0 * -inf
```

Using a large finite negative value for unselected rows might avoid the `NaN`,
but it would change cchuter's mask polarity/magnitude and should not be slipped
into the correctness-first port without explicit numeric validation.

### Current policy

Do not wire a compute path that depends on CPU `ggml_set_rows` for DSV4
compressed masks. Continue with indexer-score/top-k graph construction, but
stop before compressed-mask materialization until one of these is true:

1. `ggml_set_rows` CPU support is fixed and covered by a small backend test,
2. an exact primitive decomposition for scatter-to-`-inf` masks is found, or
3. a deliberate finite-mask approximation is evaluated against cchuter logits
   and accepted as a documented semantic change.
