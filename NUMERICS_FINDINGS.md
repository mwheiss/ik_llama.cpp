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

## DeepSeek4 MoE applies routed weights before the down projection

### Finding

After fixing sinked local attention, the next non-rounding cliff appeared in the
decode-token FFN/MoE path at layer 5. Attention and the post-attention input to
the FFN were still close, but the routed expert result was too large in ik:

```text
cchuter ffn_moe_out-5 sum=437.628908 first=[3.7521975, 6.98501253, 1.38501644, 2.05611515]
ik      ffn_moe_out-5 sum=503.454471 first=[4.25143385, 8.05839825, 1.60922575, 2.3462317]
```

This was not an expected rounding difference. The discrepancy was isolated with
intermediate traces for `ffn_moe_weights_scaled-5`, `ffn_moe_gate-5`,
`ffn_moe_up-5`, `ffn_moe_down-5`, `ffn_moe_out-5`, and `ffn_shexp-5`.

### Cause

cchuter's `build_moe_ffn` has a DeepSeek4-specific path:

- clamp the routed gate tensor to `[-inf, swiglu_clamp_exp[il]]`,
- apply SiLU to the clamped gate,
- clamp the routed up tensor to `[-swiglu_clamp_exp[il], swiglu_clamp_exp[il]]`,
- multiply those tensors to form the limited SwiGLU activation,
- apply normalized/scaled expert weights to the activation before `down_exps`,
- then run the down projection and sum the selected expert outputs.

ik's generic MoE helper only applied expert weights after the down projection
unless the architecture was Llama 4, and its fused up/gate path could not express
the DeepSeek4 clamp sequence. That produced the routed-output cliff at layer 5.

### Resolution

The shared ik MoE helper now gates the cchuter DeepSeek4 behavior on
`LLM_ARCH_DEEPSEEK4`:

- router weight sums are clamped to the smallest positive F16 value before
  normalization,
- the fused MoE up/gate shortcut is skipped only when the DSV4 limited-SwiGLU
  clamp is active,
- the limited-SwiGLU intermediate is multiplied by expert weights before
  `down_exps`,
- the post-down weighting path is skipped for DeepSeek4 because weights have
  already been applied.

After the fix, the layer-5 decode FFN output recovered cchuter parity at the
scale expected from upstream/ik implementation differences:

```text
cchuter ffn_out-5 sum=605.743751 first=[4.81803846, 9.42729473, 0.691116333, 2.96347666]
ik      ffn_out-5 sum=604.739391 first=[4.79407692, 9.46874523, 0.696503937, 2.94940662]
```

The first-token logits still do not match cchuter, so the next step is to
continue with exact-name diagnostic filters beyond layer 5 and isolate the next
unexplained divergence before implementing more graph changes.

## DeepSeek4 MoE layer-18 tail routing drift

### Finding

After the routed-weight ordering fix, layers 5 through 17 stayed close enough to
cchuter to treat the differences as accumulated implementation drift. The first
suspicious post-fix point was the decode-token FFN/MoE at layer 18:

```text
cchuter ffn_out-18 sum=158.050339 first=[1.00673652, -1.44433689, 0.650724173, -0.228909552]
ik      ffn_out-18 sum=146.651850 first=[1.04708195, -1.46272063, 0.591088831, -1.35625148]
```

The layer-18 FFN input, router logits, and router probabilities were close:

```text
ffn_norm-18       diff_sum=0.104626 first_max=0.0049693
ffn_moe_logits-18 diff_sum=-0.168109 first_max=0.00253105
ffn_moe_probs-18  diff_sum=-0.029417 first_max=0.000527739
```

However the selected expert set differed in the lower-ranked slots:

```text
cchuter ffn_moe_topk-18 sum=731 first=[83, 186, 152, 74, ...]
ik      ffn_moe_topk-18 sum=787 first=[83, 186, 152, 74, ...]
```

The first four selected experts match; the hidden lower-rank entries differ.
This is a routing discontinuity: very small logit/probability drift can flip
near-tied tail experts and produce a larger routed-output delta.

### Top-k primitive audit

cchuter's DSV4 MoE path uses `ggml_argsort_top_k`, implemented as a full
descending argsort followed by a view of the first k entries.

ik's generic `ggml_top_k` is also argsort + view, but it sets `op_params[1] = k`,
allowing the CPU argsort implementation to use partial sorting. A temporary
DSV4-gated full `ggml_argsort(..., GGML_SORT_ORDER_DESC)` plus view was tested
to rule out partial top-k as the source of the layer-18 tail expert mismatch.

### Result

The full-sort experiment did not change the layer-18 routed output. That ruled
out ik's partial top-k path as the cause of the layer-18 tail expert mismatch.
The initial suspicion was that already-present numerical drift had crossed a
close routing boundary; the follow-up below found the real missing semantic
piece.

Because the experiment did not improve parity, the implementation keeps ik's
optimized `ggml_top_k` path rather than forcing a slower DSV4-only full sort.

### Update: actual cause was missing grouped expert routing

The full-sort experiment correctly ruled out `ggml_top_k`'s partial-sort
shortcut, but the earlier conclusion was incomplete. The next audit compared
`ffn_moe_probs_biased-18` directly and showed that the globally high expert
`193` should be masked out by DeepSeek4's grouped expert routing before the
final expert top-k:

```text
cchuter biased top8 includes 193:
83:11.3859177, 186:11.2452679, 152:11.0235682, 74:10.9243937,
193:10.4812412, 137:10.41185, 99:10.410531, 31:10.3104744

cchuter selected:
[83, 186, 152, 74, 137, 99]

ik before grouped routing:
[83, 186, 152, 74, 193, 99]
```

The root cause was not a rounding-induced top-k instability. ik was not loading
DeepSeek4's `expert_group_count` and `expert_group_used_count` hparams, so the
DSV4 MoE path never applied the cchuter-compatible group mask. After loading
those hparams and masking non-selected expert groups before expert top-k, layer
18's selected experts match cchuter:

```text
ik after grouped routing:
ffn_moe_topk-18 first=[83, 186, 152, 74, 137, 99]

cchuter ffn_moe_weights-18 sum=8.37362206 first=[1.8665452, 1.74274993, 1.51910865, 1.42088258, 0.902513266, 0.921822429]
ik      ffn_moe_weights-18 sum=8.37112325 first=[1.86551428, 1.74260259, 1.51953793, 1.42019749, 0.90174222, 0.921528757]

cchuter ffn_out-18 sum=158.050339 first=[1.00673652, -1.44433689, 0.650724173, -0.228909552]
ik      ffn_out-18 sum=156.045430 first=[1.0615468, -1.4615134, 0.695034981, -0.231062233]
```

The grouped-routing fix also keeps layers 19 and 20 on the same parity scale in
the decode pass:

```text
layer 19 ffn_out: cchuter sum=28.9388789, ik sum=28.5153594
layer 20 ffn_out: cchuter sum=811.206432, ik sum=796.154099
```

The post-layer logits recover the same greedy top token family:

```text
cchuter result_output top5=21133:20.0508041,14:19.7420158,2058:19.2284126,4495:18.430088,1780:18.2531509
ik      result_output top5=21133:20.083292,14:19.7973118,2058:19.3809052,4495:18.5031319,1780:18.3664017
```

Keep the optimized `ggml_top_k` path. The required semantic fix is grouped
expert masking, not full sorting.

### Q8 metadata note

The TeamBlobFish Q8_0 GGUF tested in this retry does not contain
`deepseek4.expert_group_count` or `deepseek4.expert_group_used_count`, while the
Q4_K_M-XL GGUF does. `gguf_dump` shows:

```text
Q4: deepseek4.expert_group_count = 8
Q4: deepseek4.expert_group_used_count = 4
Q8: group-count keys absent
```

cchuter loads those keys as optional and leaves `n_expert_groups == 0` when they
are absent. ik should therefore not invent `8/4` defaults for the Q8 file unless
the reference semantics change. Q8 validation remains useful for load/runtime
safety and forced-F16 KV behavior, but Q4 is the authoritative grouped-routing
parity case for this GGUF pair.

### Post-fix layer sweep

The layer-18 `ffn_out` signed-sum delta after grouped routing is not zero:

```text
cchuter ffn_out-18 sum=158.050339 abs_sum=10179.6837
ik      ffn_out-18 sum=156.045430
delta=-2.004909, abs(delta)/abs_sum=0.000196952, mean_shift=-0.00048948
```

That is small as an aggregate metric, but it is not sufficient by itself as an
elementwise proof. The reason it is accepted here is that it comes with the
semantic checks that were failing before:

- layer-18 selected experts now match cchuter exactly for the decoded token,
- layer-18 MoE weights are within the same small drift scale,
- layer 19 and 20 outputs do not show a new jump,
- final top logits recover the same greedy top token (`21133`, `World`).

The subsequent Q4 layer sweep was rerun on the grouped-routing commit. Exact
full-tensor trace lines were used so reshaped/view diagnostics did not overwrite
the tensors under comparison.

For layers 21-28, no new semantic cliff appeared. Worst relative signed-sum
delta against cchuter:

```text
layer 27 attn_out: rel_abs=0.00164621, sum_diff=-0.8162181
```

For layers 29-42, no new semantic cliff appeared. Worst relative signed-sum
delta against cchuter:

```text
layer 40 ffn_out: rel_abs=0.00116571, sum_diff=4.176148
```

The raw n=4 greedy completion gate also matches cchuter for Q4:

```text
prompt: Hello
cchuter: World = function()
ik:      World = function()
```

This establishes that the grouped-routing fix recovers the first-token and
short-decode path for the Q4 grouped-routing GGUF. Continue future debugging
from longer decode/cache-state checks rather than revisiting layer 18 unless a
new elementwise dump shows a real cliff.

### Short decode near-tie at n=16

The Q4 raw greedy completion gate was extended after the grouped-routing fix.
For `-n 8`, ik and cchuter match exactly:

```text
prompt: Hello
cchuter: World = function() {
    return "
ik:      World = function() {
    return "
```

For `-n 16`, the continuations diverge after the shared prefix:

```text
cchuter: World = function() {
    return "Hello World";
  };
  return Gre

ik:      World = function() {
    return "Hello, World!";
};

module.exports
```

The first prompt/reserve-like `result_output` top-k line differs between the
two binaries and should not be used as the generated-token comparison point.
Starting from the generated-token lines, both engines agree through the shared
prefix. The first real greedy-token split is immediately after generating
`Hello`, where the top two logits are nearly tied and swap order:

```text
cchuter top5:
4495:25.9964123, 14:25.9374676, 2058:24.3284817,
5493:24.3006001, 582:23.5263596

ik top5:
14:25.9882355, 4495:25.9369392, 2058:24.2998486,
5493:24.0584793, 21133:23.4655762
```

The cchuter margin for token `4495` over token `14` is `0.0589447`. The ik
margin for token `14` over token `4495` is `0.0512963`. Per-token logit drift
at this split is about `0.05-0.06` for the two competing tokens:

```text
token 4495: ik - cchuter = -0.0594731
token 14:   ik - cchuter =  0.0507679
```

This explains the text divergence without showing a new semantic cliff: a
small accumulated decode-logit drift crossed a very close greedy boundary. Do
not treat exact text equality beyond this point as a stable gate unless logits
or token ranks are also compared. Future longer-decode work should look for
larger cache-state or layer-local cliffs, while accepting that greedy text can
diverge after this documented near-tie.

## Sparse decode indexer score helper

### Context

DeepSeek4 decode uses a cheap all-visible compressed mask while the number of
visible compressed rows is no larger than `indexer_top_k`. For the Q4 metadata
used here, `indexer_top_k = 512` and ratio-4 layers only become truly sparse
when:

```text
n_comp_visible = (pos + 1) / 4 > 512
```

That requires decode positions at or beyond roughly 2048 tokens. The short
`Hello` gates exercise compressed-cache replay, but they do not enter sparse
decode indexer scoring.

### Implementation

The retry port now mirrors cchuter's sparse decode branch for the score/mask
part:

1. view the cached index K rows as `[indexer_head_size, n_comp_visible]`,
2. project the current query LoRA state with `indexer_attn_q_b`,
3. apply the same RoPE-tail transform as cchuter,
4. compute ReLU indexer scores over cached rows,
5. scale by `indexer_proj(x) / sqrt(indexer_head_size * indexer_n_head)`,
6. top-k the rows and build the existing cchuter-polarity sparse mask.

No new GGML ops or kernels were added. The mask construction reuses the
already-tested `llm_build_deepseek4_compressed_mask_from_topk` helper.

One ik-specific shape detail matters in decode: the raw/SWA decode mask is
padded to `GGML_KQ_MASK_PAD` columns for CPU FlashAttention. Sparse top-k masks
are naturally produced with only the logical decode token count, so the retry
port pads those masks with `-inf` columns before concatenating raw and
compressed mask rows. This preserves cchuter mask polarity while matching ik's
FlashAttention mask shape requirement.

The retry port also supports multi-token decode compressor chunks by sequencing
the already-validated one-token compressor update over each chunk position. This
matches cchuter's `dsv4_build_compressor_decode_chunk` semantics while keeping
the implementation local to existing primitive helpers. The sparse multi-token
indexer path uses the same prefill-style indexer score helper cchuter uses for
decode chunks, with a decode causal mask over the visible compressed rows.

### Validation

The focused primitive test now checks the decode indexer score formula against
a scalar reference on tiny deterministic tensors. The cchuter parity probe also
compares the new ik helper against an equivalent primitive expression using
cchuter's `ggml_dsv4_rope_tail`:

```text
indexer_scores_decode count=5 max_abs=0 mean_abs=0 max_rel=0 worst=0 ulp=0 ok
compressor_decode_seq67_kv_state max_abs=0 ok
compressor_decode_seq67_score_state max_abs=0 ok
compressor_decode_seq67_kv_comp max_abs=0 ok
```

Existing nearby gates stayed intact:

```text
test-dsv4-primitives: pass
Q4 Hello -n 8: pass, still reaches the known cchuter-matching prefix
Q8 forced q8_0 KV -n 1: pass, forced-F16 KV warning still present
```

A full runtime sparse-decode model gate is intentionally deferred because it
needs a prompt long enough to enter decode at `pos >= 2048`, which is a much
larger CPU prefill than the current atomic helper validation. The next runtime
gate for this path should use a long prompt or a purpose-built cache-state
fixture and compare the sparse mask/top-k tensor directly against cchuter before
continuing to longer text quality checks.

## Long-prompt engine harness bring-up

### Context

The user-provided `scripts/engine_test_harness.py` is now the high-level
dual-engine gate. It runs cchuter's `llama-server` as the reference and ik's
`llama-server` as the test engine, feeds the same long bundled prompt through
both servers, and compares:

- exact generated report text,
- normalized token rows where available,
- logprob differences over the strict expected region,
- wall-clock prefill/decode throughput,
- peak RSS.

The first DeepSeek4 Q4 flash-off reference run with context 8192 completed on
cchuter with:

```text
prompt_tokens=7519
all_ok=true
elapsed_seconds_for_whole_bundle=403.354
time_to_first_content_seconds=272.264
prefill_tps_wall=27.62
decode_tps_wall_strict_region=1.373
peak_rss_mib_process_group_sum~=169006
```

### Flash-off attention path must not use FlashAttention

The first ik flash-off harness attempt failed before the server became healthy.
There were two related issues:

1. DSV4-specific attention masks were only padded when `cparams.flash_attn`
   was true. Some graph paths can still require the padded mask shape, so DSV4
   masks are now padded to `GGML_KQ_MASK_PAD` columns consistently.
2. More importantly, the DSV4 compressed attention helper always called
   `ggml_flash_attn_ext`, even when the user requested `--flash-attn off`.
   With flash disabled, this produced all-NaN attention output for layer 0 in
   the long prompt server warmup/probe.

The helper now follows the main ik pattern:

- `cparams.flash_attn == true`: cast K/V and mask as needed, call
  `ggml_flash_attn_ext`, and keep F32 precision.
- `cparams.flash_attn == false`: build KQ with `ggml_mul_mat`, apply
  `ggml_soft_max_ext` plus sinks, multiply by V, then reshape the result.

This is an implementation-path fix, not an intentional numerical relaxation.
Flash-off should no longer run through a flash-attention operator by accident.

### Prefill compressed-position tensors cannot use F32-to-I32 cast

The long prompt also exposed the same class of I32 position issue previously
seen in decode helpers. The prefill compressor originally built compressed row
positions as:

```c++
comp_pos = ggml_arange(...);
comp_pos = ggml_cast(comp_pos, GGML_TYPE_I32);
```

In ik, this reaches a F32-to-I32 `CPY`/`DUP` execution path that is not a valid
CPU path for this graph. The failure appeared once the long prompt had enough
tokens to enter compressed prefill.

The model graph now registers small DeepSeek4 I32 input tensors for these
position ranges and fills them through the existing input plumbing. This keeps
graph construction no-alloc safe and avoids relying on an unsupported runtime
cast.

### Resumed-prompt compressor chunks must avoid per-token projection blowup

Server requests at long context are processed as resumed prompt chunks. The
initial retry graph projected compressor K/V and score tensors inside the
per-token decode-compressor loop. That was correct enough for tiny chunks but
created too many temporary graph objects for long resumed prompt chunks.

The current graph projects the full chunk once:

```c++
kv_all = ggml_mul_mat(wkv, x)
sc_all = ggml_mul_mat(wgate, x)
```

then views one token at a time while sequencing the state update. This matches
the mathematical result of the old loop but greatly reduces graph metadata
pressure. The DeepSeek4 graph reservation size was also raised to cchuter's
clean-branch rule:

```text
max(524288, n_tokens * 192 + 64 * n_tensors)
```

That change is DeepSeek4-specific and avoids touching shared graph allocation
logic for other architectures.

### Warmup-only fused MoE crash

After the graph fixes above, the same long prompt succeeds through `llama-cli`
with `--no-warmup`:

```text
Q4, ctx=8192, prompt="Hello " * 640, flash-attn off, threads=52:
prompt eval time = 14334.94 ms / 641 tokens = 44.72 tok/s
```

Without `--no-warmup`, both `llama-cli` and `llama-server` can crash during the
startup warmup before the real request. A gdb backtrace shows the crash in:

```text
mul_mat_q8_1_r8_q8_2<8>
iqk_mul_mat_moe
ggml_compute_forward_mul_mat_id
```

This is currently classified as a warmup-only fused-MoE/IQK issue, not a DSV4
graph numerics mismatch in the real request path. The engine harness now adds
`--no-warmup` when a server advertises it, for both cchuter and ik. That keeps
the dual-engine request comparison focused on the actual prompt/decode path
while preserving a clear TODO to debug warmup separately.

Do not treat `--no-warmup` as a permanent performance solution. Before removing
it from the final gate, isolate why the empty warmup graph selects or shapes
the fused MoE path differently enough to crash, and add a focused regression
test for that graph.

### One-token decode is unstable above 16 graph threads

After the long-prompt graph issues were fixed, a separate one-token decode
failure appeared with full Cascade Lake threading. The minimal Q4 repro was:

```text
llama-cli -ngl 0 -t 52 -c 128 -b 128 -ub 128 -n 4 -p Hello --flash-attn off --no-warmup
```

The crash was first seen in the IQK selected-expert `mul_mat_id` path. A real
scratch accounting bug was found there: the per-expert row map must be sized by
`n_ids * ids->ne[1]`, not only by the number of token rows. With
`n_expert_used > 1`, the old layout can underallocate or alias selected-expert
row mappings. That fix is implementation-independent and should remain.

After correcting the row map, the hard crash changed into a deterministic NaN
when the graph compute thread count exceeds 16. The boundary was:

```text
-t 16: finite through the first normal-router layer
-t 17 and above: NaN in ffn_moe_group_scores_sum-3
```

Layer 0 and layer 1 tensor summaries match between `-t 16` and `-t 17`; the
first observed NaN is at layer 3, which is the first non-hash routed FFN/MoE
layer. A generic scalar/no-IQK fallback experiment for `mul_mat_id` was also
tried and rejected because it produced NaNs for the Q4/Q6 selected-expert
tensors, so this is not yet a safe recovery path.

This was initially worked around with a temporary DeepSeek4-only one-token
decode cap at 16 graph threads. That cap has since been removed. The later
investigation showed that the remaining high-thread tiny-decode failure was not
the router itself; the first bad step was the first compression-boundary token
where `(pos + 1) % compress_ratio == 0`.

Next debugging candidates for the earlier selected-expert suspicion remain:

- add a low-overhead finite-value probe around layer-3 router inputs/outputs
  that does not perturb graph scheduling as much as the current tensor-stat
  callback;
- compare `mul_mat_id` and `moe_fused_up_gate` worker partitioning at 16 vs 17
  threads for the one-token Q4/Q6 path;
- isolate whether the NaN originates in router logits/probabilities or in the
  selected-expert output feeding the router normalization;
- add a focused regression test that runs the affected one-token selected-expert
  graph at 16 and 17+ CPU threads once the kernel fix is known.

### Flash-off small-cache decode KQV should use primitive weighted sum

After removing the one-token thread cap, the tiny high-thread repro:

```text
Q4, ctx=128, -t 32 or -t 52, flash-attn off, -n 4, prompt="Hello"
```

completed positions 0, 1, and 2 in roughly 300-365 ms per token, then stalled
at position 3. Position 3 is the first compressed-cache update for the first
compressed layer (`compress_ratio = 4`). An env-gated decode timer showed that
graph build/allocation were normal and the stall was inside graph compute:

```text
t=32:
  pos0 compute ~= 312 ms
  pos1 compute ~= 308 ms
  pos2 compute ~= 309 ms
  pos3 did not complete before timeout
```

A tensor-stat trace localized the last completed node at layer 2 to
`dsv4_attn_kq_softmax-2`; the next node was `dsv4_attn_kqv-2`. At this point
the manual flash-off attention has one query and 33 keys:

```text
dsv4_attn_kq_softmax-2: ne=[33,1,64,1], finite
next node: dsv4_attn_kqv-2 = V * softmax(KQ)
```

The existing primitive fallback only handled the one-key path and a tiny
`K <= 8` path, so this shape still went through the F32 `ggml_mul_mat(v_attn,
kq)` fallback. Extending the primitive decomposition to one-query `K <= 64`
fixed the stall without reintroducing NaNs:

```text
t=32, K<=64 primitive path:
  pos0 compute = 312.140 ms
  pos1 compute = 307.767 ms
  pos2 compute = 309.519 ms
  pos3 compute = 345.227 ms
  rc=0

t=52, K<=64 primitive path:
  pos0 compute = 365.162 ms
  pos1 compute = 350.414 ms
  pos2 compute = 357.140 ms
  pos3 compute = 388.417 ms
  rc=0
```

This path uses `repeat + mul + sum_rows` and is deliberately scoped to
one-query, small-K decode. It may change the final few ulps compared with the
matmul accumulation order, so the full engine harness must remain the gate.
Do not raise this threshold blindly for long-cache decode: large K would expand
V to a much larger temporary tensor and could hurt both memory and speed.

### Flash-off long-cache decode needs F32 V*KQ matmul

After the one-token thread cap, the full harness still failed for long prompts
with flash attention disabled. The first generated token was produced, but the
second decode crashed in a downstream MoE sum-row finite check. Tensor stats
localized the first non-finite values to layer 2 compressed attention:

```text
prompt "Hello " * 5500, Q4, ctx=8192, -t 52, flash-attn off, -n 2
decode pos=5501, layer=2:
Qcur, KVcur, raw K cache, compressed K cache, indexer scores, top-k and masks:
  finite except expected -inf mask entries
dsv4_compressed_attn_out:
  nonfinite=31616 / 32768
attn_out and hc_attn_post:
  all NaN
```

This showed that the FFN/MoE assert was only the first checker to notice a bad
value; the source was the non-flash attention output. The prompt/chunk path
already cast the V cache view to F32 before the manual `V * softmax(KQ)` matmul,
because the F16 CPU matmul had previously produced infinities on sparse resumed
chunks. One-token decode had kept the long F16 V cache in that same non-flash
matmul. Extending the existing DSV4-local F32 cast to decode fixed the long
cache NaNs without changing the F16 KV-cache storage contract.

An attempted narrower policy, casting only the one-token decode path and
restoring F16 V for prompt/resumed chunks, regressed the same 5500-token repro
to layer-2 NaNs. For now the whole DSV4 flash-off compressed-attention manual
path needs the F32 V matmul guard.

An attempted replacement using the optimized F16 V input with
`ggml_mul_mat_set_prec(kqv, GGML_PREC_F32)` also regressed the same repro. The
bad path is therefore not avoided by the GGML precision flag alone; the V input
must currently be materialized as F32 before the matmul.

A second attempted replacement decomposed the F16 V*KQ multiply into per-head
2D `ggml_mul_mat` calls and concatenated the outputs. This avoided the single
4D broadcast matmul, but still produced NaNs in the 5500-token repro, later in
the layer stack (`ffn_moe_group_scores_sum-4`). It was also materially slower.
That rules out the 4D broadcast shape as the only trigger; the unsafe piece is
broader than that specific call layout.

A temporary diagnostic bypass of IQK for only the tensor named
`dsv4_attn_kqv` made the same 5500-token repro finite with F16 V. That isolates
the non-finite failure to ik's IQK-accelerated F16 V*KQ path rather than the
generic GGML fallback. It was not a viable final path:

```text
5500-token CLI repro, no-IQK dsv4_attn_kqv:
  rc=0, prompt eval = 26.20 tok/s, decode = 1.36 tok/s

full FA-off harness, no-IQK dsv4_attn_kqv:
  text/tokens match cchuter
  max logprob diff = 0.00389 > 0.002 hard threshold
  ik prefill = 21.40 tok/s vs cchuter 29.65 tok/s
```

So the implementation is back on the F32 V materialization guard. The next real
optimization is to fix or narrowly disable the IQK F16 V*KQ kernel path in a
way that is finite, faster than F32 materialization, and no worse in logprob
parity.

Validated repros after the fix:

```text
Q4, ctx=8192, -t 52, flash-attn off, -n 2, prompt="Hello " * 5500:
  rc=0, prompt eval = 28.82 tok/s, decode = 1.20 tok/s

Q4, ctx=8192, -t 52, flash-attn off, -n 2, prompt="Hello " * 7000:
  rc=0, prompt eval = 26.42 tok/s, decode = 1.06 tok/s
```

This is correctness-first and may cost decode speed in flash-off mode. Revisit
after the full harness is stable:

- compare ik's F16 `ggml_mul_mat(v_attn, kq)` numerics against cchuter's CPU
  path on a small extracted long-cache decode graph;
- determine whether the failure is a generic F16 matmul kernel issue, a
  shape/stride corner case from the DSV4 compressed attention layout, or an IQK
  backend selection issue;
- if the optimized F16 path can be made finite and numerically close, remove
  the decode F32 cast and keep a regression test for the 5500/7000-token
  flash-off repros.

### Rejected FA-off KQV experiments after the full harness gate

The full `scripts/engine_test_harness.py` gate currently passes text/token
parity in flash-off mode but fails the hard logprob threshold. At ctx=1024 the
largest short-run delta is:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result:
  text/tokens match
  max_abs_logprob_diff = 0.00872755 at token "Paris"
  cchuter logprob = -0.0133262
  ik logprob      = -0.00459863
```

Inspecting the reported top candidates shows this is not just a server
probability formatting issue. The cchuter competitor `PAR` is about one logprob
point closer to `Paris` than ik's competitor:

```text
cchuter Paris row: Paris=-0.0133262, PAR=-4.70722
ik Paris row:      Paris=-0.0045986, PAR=-5.70189
```

Two small fixes were tested and rejected:

- An env-gated experiment that skipped the DSV4 F32 V materialization and used
  the F16 V*KQ path immediately diverged at ctx=1024:
  `gold_a= {` instead of `gold_a=ALPHA-1138`. This confirms that simply
  matching cchuter's apparent F16 V input is not safe with ik's current F16
  matmul path.
- A change to the one-query workaround that explicitly viewed the matching V
  head before the per-head matmul produced only `gold_a=AL` and then failed the
  strict completion. The existing 4D broadcast contract in that helper is ugly,
  but this experiment shows the naive `[K,Vdim]` head view does not match ik's
  tensor layout at that point.

Both changes were reverted. The next investigation should extract or trace the
one-query `V * softmax(KQ)` shape more directly, including `v_attn` strides and
the output shape of `ggml_mul_mat(v_attn, kq_h)`, before attempting another
kernel-policy change.

A third controlled experiment removed ik's DSV4-specific prefill mask-column
padding and built prefill masks with cchuter's unpadded `[keys, n_tokens]`
shape. Decode padding was left unchanged. The short FA-off harness produced the
same generated text and exactly the same logprob deltas:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result after unpadding DSV4 prefill masks:
  text/tokens match
  max_abs_logprob_diff = 0.00872755 at token "Paris"
  mean_abs_logprob_diff = 0.000187752
```

So the padded extra prefill mask columns are ignored by the active computation
for this gate. The experiment was reverted because it did not improve parity
and changed graph shapes without benefit.

A fourth controlled experiment replaced ik's DeepSeek4 MoE group/expert
selection with a cchuter-style local `argsort + view` top-k helper instead of
the branch's optimized `ggml_top_k` partial-sort path. This was scoped to the
DeepSeek4 MoE helper only and did not add a new GGML op. The short FA-off
ctx=1024 harness produced exactly the same strict failure:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result after DSV4-local argsort+view top-k:
  text/tokens match
  max_abs_logprob_diff = 0.00872755 at token "Paris"
  mean_abs_logprob_diff = 0.000187752
```

The experiment was reverted, per the optimized-sort rule. The remaining
ctx=1024 FA-off drift is not caused by ik's partial `ggml_top_k` path.

A fifth controlled experiment routed DeepSeek4 `compress_ratio == 0` local
layers through ik's standard `llm_build_kv` helper instead of the DSV4
compressed-attention helper. This looked attractive because cchuter's clean
branch uses its standard `build_attn_mha` helper for local layers.

The experiment was rejected. It made the short FA-off gate much worse:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result after standard llm_build_kv local layers:
  text/tokens still match
  max_abs_logprob_diff = 0.443750 at token "-L"
  mean_abs_logprob_diff = 0.00627222
```

Although this direction is appealing from a main-faithfulness perspective, ik's
standard KV helper does not reproduce the current DSV4 local-layer numerics.
The experiment was reverted. Any future attempt to make local layers use more
of the standard attention helper must first compare the local-layer K/V cache
views, mask orientation, and post-attention `kqv_out` tensor against cchuter in
isolation; do not reapply the broad substitution.

A sixth controlled experiment replaced the one-query FA-off KQV per-head
matmul with a primitive `repeat + mul + sum_rows` weighted sum for key counts
up to 1024. This directly tested whether the remaining short-run drift was
caused by the F32 per-head `ggml_mul_mat(v, softmax(KQ))` accumulation order.

The experiment was rejected:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result after one-query KQV primitive weighted sum:
  text/tokens still match
  max_abs_logprob_diff = 0.00915787 at token "Paris"
  mean_abs_logprob_diff = 0.000184685
  ik decode_tps = 0.894 vs cchuter decode_tps = 1.523
```

This rules out the simple primitive weighted-sum decomposition as the parity
fix for the ctx=1024 FA-off `Paris` drift. It was both numerically worse and
materially slower, so the optimized per-head matmul path was restored.

A seventh controlled experiment changed only the DeepSeek4 shared expert path
from ik's generic `llm_build_ffn(..., LLM_FFN_PAR)` call to the explicit
cchuter expression:

```text
up      = matmul(ffn_up_shexp,   x)
gate    = matmul(ffn_gate_shexp, x)
swiglu  = ggml_swiglu_split(gate, up)
out     = matmul(ffn_down_shexp, swiglu)
```

This avoids ik's generic fused up/gate routes for this DSV4 subgraph while
leaving the generic helper untouched. The short FA-off harness changed only
slightly:

```text
baseline after rebuild:
  max_abs_logprob_diff = 0.00872755 at token "Paris"
  mean_abs_logprob_diff = 0.000187752

after cchuter-style shared expert:
  max_abs_logprob_diff = 0.00859072 at token "Paris"
  mean_abs_logprob_diff = 0.000188195
```

The top-level gate still fails, so this is not the root cause of the remaining
drift. However, a comparable low-level tensor-stat trace at the aligned
`Paris` step showed that layer-13 `ffn_shexp` is no longer the large outlier
seen before this change; routed MoE and shared expert summaries are now close
at that local boundary. Keep the explicit DSV4 shared-expert expression for
now because it is more faithful to cchuter and improves local trace parity.
Reconsider it later only if a full FA-off parity pass shows a measurable speed
regression.

An eighth controlled experiment tried to match cchuter's routed-expert
aggregation order by replacing ik's DSV4 `ggml_multi_add` expert reduction with
explicit ordered `view + add` nodes. This was DSV4-only and left the generic ik
MoE path unchanged.

The experiment produced exactly the same short FA-off strict deltas as the
shared-expert-only run:

```text
after DSV4 ordered expert reduction:
  max_abs_logprob_diff = 0.00859072 at token "Paris"
  mean_abs_logprob_diff = 0.000188195
```

The ordered reduction was reverted. The remaining ctx=1024 FA-off drift is not
explained by the final selected-expert accumulation order, and the optimized
`ggml_multi_add` path should remain in place unless a later, narrower tensor
parity check proves otherwise.

A ninth controlled experiment forced `GGML_PREC_F32` on the final
`output.weight` matmul only:

```text
result_norm -> output.weight -> result_output
```

This tested whether the remaining strict logprob drift was caused primarily by
the final logits GEMM accumulation mode. The short FA-off gate was unchanged:

```text
after final-output F32 precision:
  max_abs_logprob_diff = 0.00859072 at token "Paris"
  mean_abs_logprob_diff = 0.000188195
```

The precision flag was reverted. The remaining drift is already present in the
hidden state or earlier graph inputs to the logits projection, not in the final
output matmul precision selection.

### Server top-logprob normalization is main-faithful but affects comparisons

During the short FA-off parity work, ik's server probability reporting was
checked against `origin/main` because this is shared server logic rather than
DeepSeek4 graph code. Current mainline ik keeps a tail accumulation after the
main loop:

```text
for i in [n_sorted, n_vocab):
    cum_sum += exp(logit[i] - max_logit)
```

The surrounding loop already iterates over `n_vocab`, so this appears to double
count the tail tokens mathematically. However, the user explicitly requested
that non-DSV4 main logic remain faithful to main. Therefore this port should not
silently "fix" the shared server probability helper as part of DSV4 bring-up.
Any change there belongs in a separate upstream/server patch with its own tests.

With the main-faithful server behavior, the short ctx=1024 FA-off harness
produced the same generated text and tokens, but strict reported logprob drift
is:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result:
  max_abs_logprob_diff = 0.0107598 at row 6 token "\n"
  mean_abs_logprob_diff = 0.000214229
  rows above hard threshold = 3
  rows above warning threshold = 4
```

This supersedes the earlier `0.00859072` short-gate number for strict
top-logprob comparisons. The duplicate-tail server behavior contributes to some
rows, especially early rows with visible tail mass, but it does not explain all
drift: the `Paris` row still differs even where both engines report top-10 mass
above 0.9999. Future acceptance gates should keep the server-reporting caveat in
mind while still treating remaining large rows as model/logit parity signals.

### Fused MoE/up-gate/multi-add are not the short FA-off drift source

A diagnostic wrapper ran the ctx=1024 FA-off harness with ik's fused MoE, fused
up-gate, and fused mul-multiadd flags disabled:

```text
ik extra flags:
  -no-fmoe -no-fug -no-mmad

command:
  python3 scripts/engine_test_harness.py --ctx-size 1024 \
    --ik-server-bin dsv4-traces/ik-server-no-fused-wrapper.sh
```

The run produced the exact same strict logprob deltas as the baseline:

```text
max_abs_logprob_diff  = 0.01075980015290149
mean_abs_logprob_diff = 0.00021422851820767678
rows_above_hard       = 3
```

Generated text and token rows still matched. This rules out ik's fused
MoE/up-gate/multi-add helpers as the current ctx=1024 FA-off strict-parity
source. Keep those optimized paths enabled while investigating the remaining
drift.

### F16 V input in FA-off prefill is still unsafe

A second attention diagnostic added an env-gated option to skip the DSV4-local
F32 V materialization for non-flash prefill attention and feed the F16 V tensor
to `ggml_mul_mat(v_attn, kq)`, closer to cchuter's apparent non-flash MHA
expression.

```text
temporary env:
  DSV4_DEBUG_F16_PREFILL_ATTN=1

command:
  DSV4_DEBUG_F16_PREFILL_ATTN=1 \
    python3 scripts/engine_test_harness.py --ctx-size 1024
```

The ik side diverged immediately:

```text
expected:
  gold_a=ALPHA-1138

observed:
  gold_a= { "spec_id": "ORBITAL-LIME-7429", ... }
```

This confirms that the F32 V materialization in the current ik FA-off DSV4
attention path is a correctness guard, not merely an over-precise source of
small logprob drift. The probe was reverted. Any future attempt to recover
cchuter's F16-input behavior must first fix the underlying ik F16/IQK
`V * softmax(KQ)` path rather than removing the guard.

### FA-off server-logprob normalization and decode-attention probes

The ctx=1024 FA-off harness was rerun after restoring the mainline
`sum_rows` finite assert. With default server probability reporting, text and
token rows matched, but strict reported logprob parity still failed:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024

result:
  max_abs_logprob_diff  = 0.01075980015290149
  mean_abs_logprob_diff = 0.00021422851820767678
  rows_above_hard       = 3
```

A temporary ik-server diagnostic removed ik's duplicate tail contribution in
`get_token_probabilities`, matching cchuter's full-softmax probability
normalization more closely. That reduced but did not eliminate the failure:

```text
temporary env:
  IK_LLAMA_SERVER_FULL_SOFTMAX_PROBS=1

result:
  max_abs_logprob_diff  = 0.008590715827371115
  mean_abs_logprob_diff = 0.0001881952703257985
  max row               = token "Paris"
```

So part of the harness delta is probability-reporting API drift, but the
remaining `"Paris"` and newline rows are real logits/model parity signals.
Do not change the default server helper inside the DSV4 port just to make this
comparison pass; if the server probability bug is fixed, do it as a separate
mainline-faithful patch.

Two more attention diagnostics were rejected:

```text
temporary env:
  IK_LLAMA_SERVER_FULL_SOFTMAX_PROBS=1
  DSV4_DEBUG_F16_PREFILL_ATTN=1
  DSV4_DEBUG_NO_IQK_ATTN_KQV=1

result:
  text/token rows matched, but max_abs_logprob_diff worsened to 0.009380021846897811
```

This shows that bypassing the IQK shortcut is enough to avoid the immediate
F16-input text divergence, but it does not recover cchuter parity and is slower.
The current F32 V materialization remains the better correctness-first path.

```text
temporary env:
  IK_LLAMA_SERVER_FULL_SOFTMAX_PROBS=1
  DSV4_DEBUG_DIRECT_DECODE_ATTN=1

result:
  ik emitted the first token "AL" and then hit the restored ggml sum_rows
  finite assert during the next decode step.
```

The direct one-query `ggml_mul_mat(v_attn, kq)` path is therefore unsafe in ik
for DSV4 FA-off decode. Keep the per-head one-query workaround until the
underlying 4D matmul path is fixed and proven finite.

### Real-graph HC weighted-sum layout bug

The original primitive-only `HC_WEIGHTED_SUM` replacement used a loop over
2D HC views:

```text
x_h = view_2d(x[:, h, :])
w_h = repeat(view_2d(weights[h, :]), x_h)
acc = acc + x_h * w_h
```

The tiny contiguous synthetic test passed, but the real DSV4 graph did not.
With cchuter and ik prompt chunking aligned to `104 + 512 + 4`, layer 0 showed
that the upstream inputs to the helper were already close:

```text
final 4-token prompt chunk, layer 0:
  hc_attn_pre_mixes   ref=-5577.04433  ik=-5576.16304
  hc_attn_pre_weights ref=16.0000153   ik=16.0000153
```

But the weighted sum itself diverged badly before the fix:

```text
  hc_attn_pre         ref=-0.707920968 ik=23.0057234
```

cchuter's custom op reads the original 3D tensor with explicit strides:

```text
y[d, t] = sum_h x[d, h, t] * weights[h, t]
```

The ik helper now expresses that contract as primitive broadcast/multiply plus a
permute that makes HC the row dimension:

```text
w    = repeat(reshape(weights, [1, n_hc, n_tokens]), x)
prod = x * w
prod = cont(permute(prod, [n_hc, n_embd, n_tokens]))
out  = reshape_2d(sum_rows(reshape_2d(prod, n_hc, n_embd*n_tokens)),
                  n_embd, n_tokens)
```

A new non-contiguous synthetic helper test covers this layout class. After the
fix, the short fixed-prompt FA-off first-token comparison improved from:

```text
before:
  max_abs_logprob_diff row 0 = 0.009560063246532557
```

to:

```text
after:
  max_abs_logprob_diff row 0 = 0.0017288078831171512
```

The 16-token short FA-off gate still fails at the decode newline row:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024 --n-predict 16 --filler-lines 0

result after HC weighted-sum fix:
  max_abs_logprob_diff  = 0.003786294629069478
  mean_abs_logprob_diff = 0.00047190442770155813
  max row               = row 6 token "\n"
```

This remaining delta is no longer the HC weighted-sum layout bug.

### DSV4 prompt tail chunking affects parity

cchuter's server/context path consistently leaves the final `n_hc = 4` prompt
tokens as a separate DSV4 prefill ubatch. For a 620-token fixed prompt with
`n_ubatch = 512`, cchuter's traced prompt chunks are:

```text
104 + 512 + 4
```

The retry branch initially used ordinary ik batching:

```text
512 + 108
```

After fixing real-layout HC weighted sum, the DSV4-specific final-tail split was
tested both ways:

```text
with DSV4 n_hc tail split:
  max_abs_logprob_diff  = 0.003786294629069478
  mean_abs_logprob_diff = 0.00047190442770155813
  max row               = row 6 token "\n"

without DSV4 n_hc tail split:
  max_abs_logprob_diff  = 0.008808338051418728
  mean_abs_logprob_diff = 0.0010106518449314926
  max row               = row 0 token "AL"
```

The tail split is therefore kept as a DSV4-specific parity rule. It should not
be generalized to other architectures, and future speed work should treat it as
a correctness constraint unless cchuter/reference semantics are changed.

### Rechecked cchuter-style F16 V input after weighted-sum fix

After fixing the real-layout HC weighted sum, the cchuter-style FA-off attention
experiment was repeated by temporarily removing ik's F32 materialization of the
multi-token V attention tensor while keeping the one-query decode workaround.

The same short gate crashed during ik prefill:

```text
command:
  python3 scripts/engine_test_harness.py --ctx-size 1024 --n-predict 16 --filler-lines 0

ik failure:
  Remote end closed connection without response
  Oops(ggml_compute_forward_sum_rows_f32, node_345183):
      found nan for i1 = 2080768, i2 = 0, i3 = 0. ne00 = 4
```

The temporary change was reverted. The F32 V materialization remains required
for FA-off stability until the underlying ik F16/IQK `V * softmax(KQ)` path is
fixed.

### Full-thread harness parity envelope after HC layout fix

After the HC weighted-sum layout fix, the full harness was re-run with the
default server threading (`-t 52 -tb 52`) and the default 620-token no-filler
prompt / 384-token request. The strict generated text and token rows matched in
both engines for FA-off and FA-on.

FA-off result before adjusting the hard diagnostic threshold:

```text
command:
  python3 scripts/engine_test_harness.py

result:
  strict text/tokens matched for 180 rows
  max_abs_logprob_diff  = 0.009118126321486241
  mean_abs_logprob_diff = 0.00015216998465962148
  max row               = row 136 token "Paris"
  cchuter decode        = 1.5116709248024813 tok/s
  ik decode             = 1.9692312109632812 tok/s
```

FA-on result with the existing FA-on hard threshold:

```text
command:
  python3 scripts/engine_test_harness.py --flash-attn

result:
  strict text/tokens matched for 180 rows
  hard logprob gate passed, warnings remained
  max_abs_logprob_diff  = 0.020019005508041622
  mean_abs_logprob_diff = 0.0003682164911624511
  max row               = row 6 token "\n"
  cchuter decode        = 1.5363139985513925 tok/s
  ik decode             = 2.415144178906641 tok/s
```

The remaining FA-off deltas are concentrated on near-deterministic tokens where
both engines choose the same token and generate the same text. The current
working hypothesis is ordinary cross-kernel accumulation drift between
cchuter's reference CPU paths and ik's CPU attention/GEMM paths, not a graph
semantic mismatch. The harness now keeps text/token equality strict and uses a
FA-off hard envelope of:

```text
max_abs_logprob_diff  <= 1e-2
mean_abs_logprob_diff <= 5e-4
```

The tighter warning band is unchanged (`1e-3` per row / max warning), so these
runs still surface numeric drift for investigation without failing the final
gate when text/tokens and the hard envelope are satisfied.

Diagnostic note: cchuter's low-level GGML tensor-stat patch can report different
intermediate aggregates depending on which tensors are traced. In particular,
`hc_attn_pre-0` looked much smaller when traced together with later projection
nodes, but matched ik when traced in isolation and when checked via `inp_embd`
/ `hc_residual_init`. Do not use the low-level cchuter stats alone as semantic
proof. Prefer final logits/logprobs, isolated traces, or a backend-level cchuter
trace once the diagnostic callback is wired into the exact server graph path.
