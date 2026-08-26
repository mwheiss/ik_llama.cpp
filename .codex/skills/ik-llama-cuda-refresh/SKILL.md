---
name: ik-llama-cuda-refresh
description: Refresh /storage/ik_llama.cpp from canonical upstream, preserve its request-scoped CUDA graph cache cleanup when upstream still lacks an equivalent lifecycle, rebuild the existing CUDA configuration, and synchronize the rebased result to the mwheiss fork when authorized. Use for recurring ik_llama.cpp update, CUDA cleanup replay, fork synchronization, or normal local rebuild requests.
---

# Refresh ik_llama.cpp

Work only in `/storage/ik_llama.cpp`. Expect this two-remote layout:

- `origin`: `git@github.com:mwheiss/ik_llama.cpp.git` (the writable fork)
- `upstream`: `https://github.com/ikawrakow/ik_llama.cpp.git` (canonical)

Report and stop if either name points elsewhere. A legacy checkout with only
canonical `origin` may be migrated by renaming it to `upstream` and adding the
fork as `origin` when the user authorizes setting up this layout.

## Preserve state

- Inspect `git status --short --branch`, remotes, branch tracking, and the local
  commits relative to both `upstream/main` and `origin/main` before changing
  anything. Record the fetched fork-main hash before any publication.
- Preserve unrelated dirty work. Do not stash, discard, push, or rewrite
  unrelated commits without explicit authorization.
- Fetch and prune both remotes before deciding what must be replayed. Updating
  the local branch is authorized only when the user's request includes an
  update, pull, refresh, or synchronization.
- Keep the CUDA cleanup as one commit and this project skill as a separate
  commit on top of `upstream/main`. Rebase them when the worktree is clean. If
  conflict resolution changes either commit, inspect the final diff and retain
  coherent atomic commits.
- Track local `main` against `origin/main`, but use explicit `upstream/main` as
  the rebase base. Never push to `upstream`.

## Decide whether cleanup is still needed

Inspect upstream commits and current CUDA graph ownership, keys, and lifecycle;
do not decide from commit subjects alone. An upstream change is equivalent only
if it either:

- provides a safe request- or context-scoped way to release cached CUDA graph
  executables, or
- bounds/evicts the graph cache so transient graph keys cannot grow it without
  limit.

Graph UIDs, stricter compatibility comparisons, capture failure cutoffs, and
destructor cleanup improve reuse or correctness but are not by themselves a
cache lifecycle strategy. If upstream is truly equivalent, omit the local patch
and cite the exact implementation in the handoff.

When replaying the local behavior, preserve these invariants unless upstream
APIs require a justified adaptation:

- Export the CUDA cleanup function from `ggml/include/ggml-cuda.h` and implement
  it in `ggml/src/ggml-cuda.cu` under `USE_CUDA_GRAPH`.
- Use the same context key passed as the third argument to
  `ggml_backend_cuda_init`; verify this live rather than assuming its type.
- Wait until graph capture is idle under the CUDA graph lock, synchronize owned
  streams, clear `cur_graph`, and destroy the scoped `cuda_graphs` entries.
- In the server, include the CUDA header only under `GGML_USE_CUDA`. Honor
  `clear_cuda_graph_cache` only for `n_parallel == 1`, synchronize before
  clearing, and handle both the target context and a distinct speculative
  companion context.

## Build and verify

Prefer the existing `build/CMakeCache.txt`; inspect its Release, CUDA, graph,
architecture, native, and server settings and rebuild without replacing them:

```bash
CCACHE_DIR=/tmp/ik-llama-cpp-ccache cmake --build build --config Release -j
```

The task-scoped cache avoids failures when the default ccache directory is not
writable. If no build configuration exists, configure a Release CUDA build with
the project's documented defaults and any user-provided hardware flags before
building.

Afterward, verify the build exited successfully, inspect the rebuilt
`build/bin/llama-server`, and confirm the worktree is clean with exactly the
intended commits ahead of `upstream/main`.

## Synchronize the fork

Push only when the user authorizes publication. A normal fast-forward push is
preferred. When replaying the cleanup changes commit identity and the fetched
fork still contains its older equivalent commit, update only `main` with an
explicit hash-pinned lease:

```bash
git push --force-with-lease=main:<observed-origin-main-sha> origin main
```

Do not use a bare `--force`, delete branches, or rewrite unrelated fork refs.
After pushing, fetch `origin` again and require `origin/main == HEAD`,
`upstream/main` to be an ancestor of `HEAD`, and zero unexpected divergence.
Report upstream, fork, and local hashes; the semantic decision about the cleanup
strategy; effective build configuration; validation performed; and exclusions.
