These full-vocabulary logits come from llama.cpp v1.0.0 at
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, using exactly the model and
adapter hashes in `reference.json`. The fixture stores F32 values in little
endian order, compressed with gzip; its raw hash and every case hash are pinned.

The source prompts and settings are the historical server golden cases: life,
joke, long Lorem ipsum, ChatML book/code, quoted France/Germany prompts, and
Shakespeare MoE LoRA off/on. The JSON records the exact token histories, slot
limits, batch sizes, F16 KV format and other context parameters. The long prompt
keeps BOS and discards three blocks of 63 tokens, yielding 112 input tokens.
Code and Lorem histories include context shifts. No candidate output is used
as the oracle.

Bounds were frozen from reference-only CPU/CUDA and F16/F32 KV controls before
measuring the corrected candidate. Each bound is twice that case's largest
reference precision deviation, with the independently established F32 floor.
Failed bounds must be investigated, not increased. Clear top-1 choices must
also agree. The original broken MoE computation fails these checks by orders
of magnitude. Hot LoRA toggles must reproduce the corresponding cold reference.

The reference was qualified by its native tests and by an independent NumPy
float64 implementation of dense and MoE LLaMA, including the LoRA matrices.
Detailed controls are retained in `reference.json` and in the local audit
`build/golden-reference-audit-20261116`.

Prepare the pinned models with `examples/server/tests/prepare_fixtures.py`.
Set `LLAMA_QUALITY_FIXTURES` to that fixture directory when running CTest.
Without it, CTest reports this gate as **skipped**. To run the GPU gate, invoke
`tests/test-model-quality.py --probe BUILD/bin/model-quality-probe --fixtures DIR --gpu`
after obtaining authorization to use the GPU.

Stochastic text is not a portable numerical oracle: ik_llama's cumulative
sampler consumes one MT19937 draw; the reference's `std::discrete_distribution`
consumes a different draw sequence. These tests compare distributions on
identical forced histories, while server tests retain replay/cache invariants
and the deterministic chat/LoRA word guards that the reference reproduces.
