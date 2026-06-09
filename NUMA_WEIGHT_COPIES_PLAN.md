# NUMA Weight Copies Plan

## Objective

Prototype an opt-in CPU runtime mode where one logical DeepSeek V4 server can
use both sockets while each socket reads local copies of read-only model
weights. The goal is to keep the prefill advantage of all physical cores without
paying the decode penalty from remote model-weight reads.

This is a backend/runtime project, not a DeepSeek4 graph change.

## Evidence

The best measured single-request latency path is a private model copy placed on
one NUMA node:

```text
node0 no-mmap, 26/26: prefill 31.1264 tok/s, decode 3.5608 tok/s, wall 70.47 s
node1 no-mmap, 26/26: prefill 30.6682 tok/s, decode 3.4311 tok/s, wall 72.68 s
```

All physical cores with an interleaved private copy improve prefill but lose
decode throughput relative to the best node-local run:

```text
all physical cores, interleaved no-mmap, 52/52:
  prefill 34.7611 tok/s, decode 3.2777 tok/s, wall 72.75 s
```

Two separate node-local `--no-mmap` servers show that replicated private model
copies can improve aggregate serving throughput when memory capacity allows it:

```text
node0 private copy: prefill 30.67 tok/s, decode 3.73 tok/s, wall 68.50 s
node1 private copy: prefill 30.07 tok/s, decode 3.55 tok/s, wall 71.33 s
aggregate decode-only after first token: 7.10 tok/s
```

Mmap/shared page-cache variants are more memory-efficient but can place shared
pages on the wrong node and make one socket read mostly remote memory. This is
why a real in-process replicated-weight implementation is worth considering.

## Non-Goals

- Do not change DSV4 graph semantics.
- Do not change quant kernels as part of the first infrastructure patch.
- Do not replicate KV cache, activations, or mutable scratch buffers initially.
- Do not add GPU/backend code.
- Do not make replication the default before Q4 and Q8 gates pass.

## Candidate Designs

### 1. Process-Level Replication

Run one server per NUMA node with `--no-mmap`, explicit CPU binding, explicit
memory binding, and a small load balancer.

Status: already validated as a deployment topology.

Pros:
- no ik code changes;
- clean memory locality;
- aggregate throughput is good.

Cons:
- doubles model memory;
- no single-prompt use of both sockets;
- one-server continuous batching is currently not a valid DSV4 baseline because
  `-np > 1` aborts in `build_deepseek4`.

### 2. Backend Buffer Replication

Extend CPU backend/model loading so selected read-only weight buffers have one
copy per NUMA node. Compute code chooses the local pointer for the current CPU
or worker thread.

Pros:
- one logical server;
- keeps one KV/cache state;
- can target only large read-only model tensors.

Cons:
- touches backend buffer lifetime and tensor data assumptions;
- needs careful thread pinning and node detection;
- doubles memory for replicated weights;
- views/repacking/overrides may assume one canonical pointer.

### 3. Matmul-Level Replicated Source Pointer

Keep tensor metadata unchanged, but attach optional per-node source pointers for
large immutable `src0` weight tensors. Matmul and `mul_mat_id` dispatch resolve
the local pointer at the start of the worker compute.

Pros:
- narrower first prototype than full backend-buffer replication;
- directly targets perf hot spots from `perf`: projection matmuls and MoE
  selected-expert matmuls;
- avoids changing graph construction.

Cons:
- still needs a place to store and free per-node copies;
- only helps kernels that are explicitly taught to use the replicated pointer;
- must prove every replicated tensor is immutable.

This is the preferred first implementation path.

## First Prototype Shape

1. Add an opt-in runtime flag or environment variable, for example
   `GGML_NUMA_REPLICATE_WEIGHTS=1`. A CLI flag can come later once the backend
   shape is proven.
2. Detect NUMA nodes and CPU to node mapping using `libnuma` where available,
   with a disabled fallback when NUMA is unavailable.
3. During model load or after tensor allocation, identify large read-only model
   tensors in CPU backend buffers. Start with quantized model weights used by
   `ggml_mul_mat` and `ggml_mul_mat_id`.
4. Allocate per-node copies with `numa_alloc_onnode`, `mbind`, or an explicit
   first-touch worker pass. Prefer `numa_alloc_onnode` for the prototype because
   it makes placement auditable with `numastat -p`.
5. Store replicated pointers in a small side table keyed by the canonical tensor
   or data pointer. Do not mutate `ggml_tensor->data` globally.
6. In CPU matmul and selected-expert matmul dispatch, resolve the local
   replicated pointer based on the worker CPU/node and pass it to the existing
   kernel path.
7. Require external or internal thread pinning for the prototype. If workers can
   migrate freely, local pointer selection becomes noisy and may hurt speed.

## Files And Areas To Inspect First

- `ggml/src/ggml-backend*`
- `ggml/src/ggml-cpu/*`
- `ggml/src/iqk*`
- `src/llama-model.cpp`
- `src/llama-load-tensors.cpp`
- DeepSeek4 MoE graph paths using `ggml_mul_mat_id`
- existing NUMA handling around the `--numa` CLI/runtime option

## Validation Gates

Each atomic patch must pass:

- `git diff --check`
- Cascade Lake CPU build
- `test-dsv4-primitives`
- VNNI artifact check
- FA-on single-thread or calibrated baseline parity
- FA-on harness against the current optimized baseline
- Q4 and Q8 smoke, including Q8 forced-F16 KV warning
- `numastat -p` evidence that replicated copies are actually node-local
- section timing and `perf` comparison against:
  - node0 no-mmap 26/26;
  - all physical cores interleaved no-mmap 52/52;
  - two-process node-local no-mmap throughput baseline.

## Failure Policy

Reject or revert the prototype if it:

- changes generated tokens under the calibrated harness;
- causes unexplained logprob drift beyond the current warning envelope;
- increases wall time versus node-local no-mmap on decode-heavy requests;
- increases memory beyond two full model copies without a clear reason;
- requires disabling DSV4 correctness assertions.

## Open Questions

- Which exact backend buffer type owns the final packed data for the hot DSV4
  Q4_K_M-XL tensors?
- Do runtime repack or IQK packing paths create derived buffers that should be
  replicated instead of the original GGUF-loaded data?
- Can per-node pointer resolution be localized to `mul_mat` and `mul_mat_id`,
  or do smaller HC/helper nodes read enough weight data to matter?
- Should the first prototype duplicate all large read-only tensors, or only the
  hot tensors identified by section timing and `perf`?
- Can ik's existing NUMA worker setup guarantee stable worker-to-node mapping,
  or do we need an explicit pinning mode before local pointer selection?
