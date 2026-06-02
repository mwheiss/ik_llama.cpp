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

## DSV4 prefill masks must be runtime inputs

### Context

DeepSeek4 uses graph-local masks whose shapes are not the same as the ordinary
KV attention mask:

- raw/local sliding-window mask,
- compressed causal/indexer mask,
- combined static raw+compressed mask.

cchuter does not treat these as compile-time constants. Its graph builder
registers them as inputs, then fills them for the current ubatch positions
before graph execution. The mask orientation is `[keys_or_rows, queries]`,
with row-major indexing equivalent to:

```text
data[iq*n0 + ik]
```

### Observed issue in the retry port

An interrupted retry step briefly replaced placeholder masks with tensors
written at graph construction time via `ggml_set_f32_nd`. That worked in a
normal allocated unit-test context, but the model graph uses a no-alloc GGML
context while building graph metadata. In the full model smoke, this approach
segfaulted before any useful DSV4 mask marker was reached.

The earlier committed placeholders were also not acceptable for runtime
correctness: they allowed the graph to reach final prefill logits, but the
masks were uninitialized rather than cchuter-equivalent.

### Current fix

The retry port now registers DSV4 mask tensors as graph inputs and fills them
in `llama_set_inputs()` from `batch.pos`:

- `RAW_WINDOW`: visible if `p0 <= p1` and the key is inside the SWA window,
- `COMPRESS_CAUSAL`: compressed row `ic` visible if
  `ic < (p1 + 1) / compress_ratio`,
- `ATTN_STATIC`: raw-window entries first, compressed-causal entries after
  the raw token block.

All entries start as `-inf`; visible entries are set to `0`. This matches
cchuter's polarity and keeps the change local to DeepSeek4-specific graph
inputs instead of modifying the shared KQ mask path.

### Validation status

After replacing graph-construction writes with runtime inputs:

- primitive parity against cchuter still passes,
- Q4 reaches final prefill logits and then the expected decode boundary,
- Q8 with requested q8_0 KV still logs the forced-F16 warning and reaches the
  same boundary,
- the prior model-smoke segfault is gone.

Future decode work must re-check these masks with nonzero positions, because
the compressed visible-row rule depends on absolute token position, not just
the query index in the ubatch.

## Compressed top-k mask scatter exposed CPU set_rows F32 bug

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

The crash was not caused by DSV4 mask semantics. The CPU `set_rows` kernel
looked up `type_traits[dst->type].from_float` and called it unconditionally.
For an F32 destination this converter is null, so the kernel jumped to address
zero. The minimal fix is to copy F32 rows directly when `dst->type` is F32 and
keep the existing converter path for non-F32 destinations.

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

The DSV4 compressed-mask helper may use `ggml_set_rows` only while the focused
synthetic mask test remains enabled and passing. That test verifies the cchuter
mask polarity:

1. selected valid rows are `0`,
2. unselected rows are `-inf`,
3. selected invalid rows are `-1e9`.

Do not replace this with a large finite unselected mask or arithmetic `-inf`
masking unless logits-level validation against cchuter shows the change is
safe.

## Decode compressor state row update uses ik SET instead of cchuter CPY view

### Context

cchuter's DeepSeek4 decode compressor updates one row of the flat compressor
state, then returns a full-state view that depends on the row-copy operation.
That state is later consumed by pooling, shifted overlap-state construction, and
state-cache storage.

### Observed ik behavior

The direct cchuter idiom:

```c++
row_view = ggml_view_2d(dst, ...);
cpy      = ggml_cpy(row_src, row_view);
state    = ggml_view_tensor(dst);
state->src[0] = cpy;
```

triggered `ggml_compute_forward_dup` aborts in the focused ik helper test when
the dependency view/copy appeared in the small standalone graph. This is not a
numeric disagreement in DeepSeek4 math; it is an ik/GGML graph plumbing
difference around destination-view CPY execution.

### Current implementation

The retry port uses ik's structured row update primitive for this CPU-only
bring-up:

```c++
state = ggml_set_2d_inplace(ctx, prev_state, row_src,
        prev_state->nb[1], row * prev_state->nb[1]);
```

The helper parity harness compares the resulting state and emitted compressed
KV against a cchuter-shaped reference for:

- ratio-4 decode position 5, where no compressed KV row is emitted,
- ratio-4 decode position 7, where the helper emits one compressed KV row and
  shifts the overlap state.

Both cases are exact in the current parity harness.

### Position tensor note

The decode helper originally constructed the single compression position as
`ggml_arange(...)->ggml_cast(I32)`. In the focused test this reached an ik
F32-to-I32 CPY path that is not suitable for this use. Decode compression uses a
known scalar position at graph-build time, so the helper now creates a direct
I32 tensor and fills it with `ggml_set_i32_1d`.

Future work can revisit cchuter's CPY-view dependency idiom if GPU or
multi-device scheduling is added to the ik DeepSeek4 path. For the current CPU
reference path, `ggml_set_2d_inplace` is numerically equivalent and covered by
cross-engine parity tests.

### No-alloc decode compression position tensor

The decode helper needs an I32 compression-position tensor only when a decode
step actually emits a compressed row. In focused tests, a direct allocated I32
tensor filled with `ggml_set_i32_1d` was the most stable parity path. In the
full model graph, however, `ctx0` is a no-alloc graph-build context. Calling
`ggml_set_i32_1d` there writes through a null tensor data pointer and segfaults
as soon as `should_compress` becomes true. For ratio-4 layers, this first
happens at decode position 3.

The first attempt to avoid the graph-build segfault used
`ggml_arange(...)->ggml_cast(I32)` for no-alloc graph contexts. That built the
graph, but execution then aborted in GGML's F32-to-I32 `dup/cpy` path, which is
not implemented for this CPU path.

The helper now keeps direct I32 tensor fills only for allocated helper/test
contexts. For no-alloc model graphs, the graph builder supplies a tiny DSV4 I32
input tensor and `llama_set_inputs()` fills it with the known compressed-row
position, e.g. `pos + 1 - compress_ratio` for one emitted row.

This keeps the existing helper parity path unchanged while making the model
graph construction safe for the first compressed-row decode step. If the
input plumbing later needs to support multi-token decode chunks, extend it to
store the corresponding vector of compressed positions instead of writing
constants into a no-alloc graph context.

## Decode compressor graph boundary is not yet runtime-position validation

The first ik graph wiring for decode compressor state updates reaches the
intentional boundary:

```text
DeepSeek V4 decode compressed cache replay not implemented yet
```

Both Q4 and Q8 smokes log the expected decode state tensor shapes for metadata
layer 2 and Q8 still logs the forced-F16 KV warning. However, the smoke reaches
this boundary with the graph-build scalar position logged as `pos=0`.

This appears to be ik's current graph-build/reserve path, not a verified
post-prefill runtime decode position. Therefore this step validates only:

- helper-level decode state math against cchuter,
- graph construction of the state update tensors,
- cache-state shape/offset plumbing,
- preservation of existing prefill-logits behavior before the boundary.

It does not yet validate real generated-token decode replay. The next cache
replay step must determine whether ik should derive the decode compressor
position from a runtime input, from `kv_head`, or from a graph-build batch
position in the same way the shared KV cache path does. Do not treat the current
`pos=0` smoke as first-token numerical correctness.

## One-token prompt prefill must not be treated as decode

### Finding

The retry graph initially used `n_tokens == 1` as the decode predicate for
DeepSeek4 compressed layers. That is wrong. cchuter's DeepSeek4 graph uses:

```c++
is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;
```

A one-token prompt such as `Hello` has `n_tokens == 1` and `pos == 0`, so it is
prefill. Treating it as decode caused ik to enter the decode compressor/cache
path before any compressed cache replay should exist.

### Fix

The ik compressed path now follows cchuter's prefill predicate. For prefill
chunks with `n_comp == 0`, it builds raw/local attention only with a DSV4
raw-window mask, matching cchuter's fallback.

This exposed a second, DSV4-local mask bug: the raw-window mask for
`n_tokens == 1` must still be padded to `GGML_KQ_MASK_PAD` when it feeds
FlashAttention. The DSV4 mask input fill code now allows `tensor->ne[1] >=
batch.n_tokens`, fills only the logical token columns, and leaves padded columns
at `-inf`.

### ik CPU FlashAttention K/V type note

After the one-token prefill path reached actual execution, ik's CPU
FlashAttention rejected composed DSV4 K/V tensors in F32:

```text
K cache f32 coupled with V cache f32 is not a supported combination on the CPU backend.
```

The shared ik attention helper already casts F32 K/V to F16 for FlashAttention,
and DSV4's KV cache is forced to F16. The DSV4 compressed attention helper now
casts composed K/V attention inputs to F16 when `cparams.flash_attn` is enabled.

### Validation

With these fixes:

- Q4 `Hello`, `-c 128 -b 128 -ub 128 -n 1`, exits 0 and produces a first token,
- Q8 with requested q8_0 KV exits 0, logs the forced-F16 warning, and produces a
  first token,
- primitive/helper parity against cchuter remains unchanged,
- patched cchuter Q4 with `--no-repack --single-turn` also exits 0 for the same
  tiny prompt.

The cchuter CLI applies different chat/single-turn prompt handling in this
smoke, so the generated text is a status sanity check only, not a logits/text
parity result.

## DeepSeek4 graph reuse must stay disabled until prefill/decode is input-driven

After the one-token prefill fix, Q4/Q8 `-n 2` initially appeared to generate
multiple tokens while logging `first_pos=0 is_prefill=1` for every one-token
graph. The cause was ik graph reuse: the graph built for the first one-token
prefill was reused for subsequent generated tokens. Since the current DSV4 graph
topology branches at graph-build time on:

```text
first_pos == 0  -> prefill/raw local fallback
first_pos > 0   -> decode/compressed cache replay
```

reusing the position-0 graph for later positions silently takes the wrong path.

The retry branch now disables graph reuse for `LLM_ARCH_DEEPSEEK4` only. This is
not a performance decision; it is a correctness guard until the DSV4 branch is
made fully input-driven or until there are separate reusable prefill/decode
graphs with compatible topology.

With DSV4 graph reuse disabled:

- Q4 `Hello -n 2` reaches `first_pos=1 is_prefill=0` and stops at
  `DeepSeek V4 decode compressed cache replay not implemented yet`,
- Q8 forced q8_0 KV reaches the same true decode boundary and still logs the
  forced-F16 warning.

## Early decode positions use raw/SWA attention before compressed rows exist

### Finding

For ratio-4 compressed DeepSeek4 layers, the first generated-token decode
positions can have no visible compressed rows yet:

```text
n_comp_visible = (pos + 1) / compress_ratio
```

At `pos=1`, this is zero for `compress_ratio=4`. Entering compressed-cache
attention composition at that point is premature; there are no compressed K/V
rows to replay. cchuter still runs the local/raw attention path for those
positions while updating the compressor state that will later emit compressed
rows.

### Current implementation

The retry port now updates the decode compressor state, writes any emitted
compressed row when one exists, and, when `n_comp_visible == 0`, runs only the
raw/SWA KV attention path for the layer. The path then applies the same
attention output un-rope, grouped output projection, and post-attention HC
expand used by the prefill compressed layer path.

This is still a staged decode implementation. It does not claim compressed
cache replay parity yet; it only handles the mathematically empty-compressed
prefix case.

### Current boundary

With Q4 `Hello -n 2`, ik reaches real decode at `first_pos=1`, runs the early
raw/SWA decode path through cache-active layers, and then stops at the explicit
NextN/tail boundary:

```text
DeepSeek V4 decode NextN/tail cache path not implemented yet
```

This boundary is useful because it distinguishes the next missing semantic
piece from a generic compressed-cache replay failure. The active layers have
DSV4 compressor/cache state scaffolding; the NextN/tail layer currently does
not. Before implementing replay for `n_comp_visible > 0`, inspect cchuter's
decode behavior for layer 42 and decide whether the tail layer needs its own
cache state, is skipped, or follows a different current-token-only path.

### Resolution: layer 42 needs DSV4 cache state

The layer-42 boundary was caused by ik's cache allocation path shortening the
KV/cache layer count by `nextn_predict_layers`, which is correct for the
mainline MTP architectures that execute their tail through a separate MTP mode
but is not faithful to cchuter's DeepSeek4 implementation.

cchuter's DeepSeek4 compressed-cache allocation iterates every metadata layer
and allocates DSV4 compressed cache/state for every layer with
`attn_compress_ratio[il] > 0`, including layer 42. The retry port now keeps
that rule local to `LLM_ARCH_DEEPSEEK4`: DSV4 uses all `hparams.n_layer` layers
for KV/cache allocation and cache-copy bookkeeping, while other non-MTP models
keep the existing `n_layer - nextn_predict_layers` behavior.

After this fix, the cache log matches the metadata inventory:

```text
DeepSeek4 compressed KV cache size = 22.18 MiB,
cache-active attn layers = 41,
cache-active indexer layers = 21
```

Q4 and Q8 forced-cache `Hello -n 2` smokes both run through layer 42 decode,
produce final logits for the second generated token, and exit successfully.
The Q8 path still logs the forced-F16 KV warning. This confirms that the
earlier 40/20 cache-active count was a bug inherited from applying generic
MTP-tail cache exclusion to DeepSeek4, not a DeepSeek4 semantic rule.

### Validation status

Primitive parity against cchuter remains unchanged after adding the early
raw/SWA decode path. The expected helper differences are still limited to
known float32 operation-order cases:

- `hc_weighted_sum`: worst 1 ULP,
- HC pre-mix/head: worst 1-2 ULP,
- grouped output F32 validation helper: tiny absolute error with worst 5 ULP.

These differences are unrelated to the early decode control flow and remain
within expected float32 rounding bounds.

## Decode compressed K/V replay cannot concat F16 cache views along dim 2

### Finding

The first compressed-cache replay implementation built the decode attention
K/V input by appending the visible compressed cache rows after the raw/SWA KV
cache rows:

```text
raw cache       [head_dim, 1, n_kv]          f16
compressed rows [head_dim, 1, n_comp_visible] f16
concat dim 2 -> [head_dim, 1, n_kv + n_comp_visible]
```

This matched the intended tensor shape, but it aborted at runtime in GGML:

```text
ggml.c:15040: GGML_ASSERT(dim == 0) failed
```

The cause was not DeepSeek4 math. In this ik/GGML version, F32 concat has a
general multi-dimensional execution path, while the fallback path used by F16
and other non-F32 types only supports `dim == 0`.

### Resolution

The DSV4 graph now promotes the two F16 cache views to F32 before the dim-2
concat. The existing flash-attention path then casts the permuted combined K/V
back to F16 before `ggml_flash_attn_ext`.

This is expected to preserve cache values exactly:

- F16-to-F32 promotion is lossless,
- the combined tensor is cast back to F16 before attention,
- the change is local to DSV4 graph construction and does not alter GGML's
  shared concat implementation.

After the fix, Q4 `Hello -c 128 -b 128 -ub 128 -n 4` exits successfully and
logs decode compressed replay through layer 42 plus final logits. Primitive
parity against cchuter remains unchanged.

## Sinked local attention must not use the IQK flash-attention shortcut

### Finding

The first Q4 `Hello` prompt comparison found a hard numerical cliff at layer 0
local attention. Before the fix, ik's `dsv4_compressed_attn_out-0` was exactly
the F16 cached `KVcur-0` repeated over 64 heads:

```text
KVcur-0                    first=[-0.625, 0.375, 0.5625, 0.25]
dsv4_compressed_attn_out-0 first=[-0.625, 0.375, 0.5625, 0.25]
```

cchuter's sinked attention for the same tensor was not equal to V:

```text
kqv_out-0 first=[-0.604744434, 0.362846643, 0.544269979, 0.241897762]
```

This is not rounding error. With a single visible K/V row, attention sinks still
participate in the softmax denominator, so the output should be V scaled by the
non-sink probability. Returning V exactly means the sink contribution was
dropped.

### Cause

ik's CPU flash-attention path can enter the IQK flash-attention shortcut before
the scalar reference path. That shortcut accepts a sinks pointer in its public
plumbing, but at least the small-cache path used by one-token DSV4 local
attention did not preserve sink semantics. The scalar CPU fallback already
handles sinks correctly.

The DSV4 helper also needed to set `op_params[4] = hparams.n_swa`, matching ik's
generic `llm_build_kv` attention helper. This keeps the sliding-window hint
available to backend attention paths.

### Resolution

The IQK flash-attention shortcut is now skipped whenever `dst->src[4]` contains
attention sinks, forcing the sink-aware scalar CPU path. DSV4's local and
compressed attention helper now also sets the SWA hint and requests F32
flash-attention accumulation.

After the fix, layer 0 aligns with cchuter within the expected implementation
drift:

```text
ik      dsv4_compressed_attn_out-0 sum=397.316409 first=[-0.604957283, 0.362974375, 0.544461548, 0.241982922]
cchuter kqv_out-0                  sum=398.910369 first=[-0.604744434, 0.362846643, 0.544269979, 0.241897762]
```

Layers 1 and 2 were also checked after this fix and stayed close at the same
scale. The first-token logits still diverge later in the graph, so the next
porting step is to continue layer-by-layer numerical comparison until the next
unexplained cliff is isolated.
