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
