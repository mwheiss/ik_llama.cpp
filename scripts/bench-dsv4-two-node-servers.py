#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_test_harness as harness  # noqa: E402


DEFAULT_MODEL = harness.DEFAULT_MODEL_DEEPSEEK
DEFAULT_SERVER = "/data/nvme/no-backup/deepseek-v4-flash/ik_llama.cpp.dsv4-port-retry/build-cpu-opt/bin/llama-server"


def jdump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def server_cmd(args: argparse.Namespace, port: int, prefix: str, threads: int, threads_batch: int) -> list[str]:
    server = str(Path(args.server_bin).resolve())
    help_text = harness.server_help_text(server)
    cmd = [*shlex.split(prefix), server, "-m", args.model]

    cmd += ["--ctx-size" if harness.has_flag(help_text, "--ctx-size") else "-c", str(args.ctx_size)]
    harness.add_if(cmd, [], help_text, "n_gpu_layers_0", ["--n-gpu-layers", "0"], "--n-gpu-layers")
    fa = harness.flash_attn_tokens(help_text, args.flash_attn)
    cmd += fa
    if args.numa:
        harness.add_if(cmd, [], help_text, "numa", ["--numa", args.numa], "--numa")
    if args.no_mmap:
        harness.add_if(cmd, [], help_text, "no_mmap", ["--no-mmap"], "--no-mmap")
    harness.add_if(cmd, [], help_text, "no_repack", ["--no-repack"], "--no-repack")
    harness.add_if(cmd, [], help_text, "no_warmup", ["--no-warmup"], "--no-warmup")
    harness.add_if(cmd, [], help_text, "single_turn", ["--single-turn"], "--single-turn")
    cmd += ["--host", harness.HOST, "--port", str(port), "-t", str(threads)]
    if harness.has_flag(help_text, "-tb"):
        cmd += ["-tb", str(threads_batch)]
    if harness.has_flag(help_text, "-np"):
        cmd += ["-np", "1"]
    if harness.has_flag(help_text, "--timeout"):
        cmd += ["--timeout", str(harness.SERVER_TIMEOUT)]
    return cmd


def numastat(pid: int) -> str:
    if not Path("/usr/bin/numastat").exists():
        return ""
    try:
        return subprocess.check_output(["numastat", "-p", str(pid)], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"numastat failed for pid {pid}: {exc}\n"


def kill_proc(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def strict_summary(response: dict[str, Any], elapsed: float, prompt_tokens: int) -> dict[str, Any]:
    rows = harness.parse_response_rows(response)
    generated = str(response.get("content", ""))
    lo, hi, _ = harness.strict_range_for_expected(rows, harness.EXPECTED_GENERATED)
    strict_tokens = 0 if lo is None or hi is None else hi - lo + 1
    time_to_first = response.get("time_to_first_content_seconds")
    decode_time = None if time_to_first is None else max(1e-9, elapsed - float(time_to_first))
    check = harness.expected_checks(generated)
    return {
        "all_ok": bool(check.get("ok") and lo is not None and hi is not None),
        "elapsed_seconds": elapsed,
        "time_to_first_content_seconds": time_to_first,
        "decode_seconds_after_first_content": decode_time,
        "prompt_tokens": prompt_tokens,
        "strict_region_tokens": strict_tokens,
        "prefill_tps_wall": None if not time_to_first else prompt_tokens / float(time_to_first),
        "decode_tps_wall_strict_region": None if not decode_time else strict_tokens / decode_time,
        "generated_prefix": generated[:160],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark two DeepSeek4 llama-server instances, one per NUMA node.")
    ap.add_argument("--server-bin", default=DEFAULT_SERVER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ctx-size", type=int, default=1024)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--filler-lines", type=int, default=0)
    ap.add_argument("--out", default="dsv4-cascade-lake-results/two-node-servers")
    ap.add_argument("--port0", type=int, default=43200)
    ap.add_argument("--port1", type=int, default=43201)
    ap.add_argument("--prefix0", default="numactl --physcpubind=0-25 --membind=0")
    ap.add_argument("--prefix1", default="numactl --physcpubind=26-51 --membind=1")
    ap.add_argument("--threads0", type=int, default=26)
    ap.add_argument("--threads1", type=int, default=26)
    ap.add_argument("--threads-batch0", type=int, default=26)
    ap.add_argument("--threads-batch1", type=int, default=26)
    ap.add_argument("--numa", default="numactl", choices=["distribute", "isolate", "numactl", ""])
    ap.add_argument("--no-mmap", action="store_true", help="Use private model loads instead of mmap/page-cache sharing.")
    ap.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=True)
    ap.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = harness.make_prompt(args.filler_lines, harness.detect_model_family(args.model))
    request = harness.make_request(prompt)
    request["n_predict"] = args.n_predict

    procs: list[subprocess.Popen[Any] | None] = [None, None]
    logs = [out / "node0-server.log", out / "node1-server.log"]
    fhs = [logs[0].open("wb"), logs[1].open("wb")]
    samplers: list[harness.MemorySampler | None] = [None, None]
    try:
        cmds = [
            server_cmd(args, args.port0, args.prefix0, args.threads0, args.threads_batch0),
            server_cmd(args, args.port1, args.prefix1, args.threads1, args.threads_batch1),
        ]
        for i, cmd in enumerate(cmds):
            (out / f"node{i}-command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
            fhs[i].write(("COMMAND: " + " ".join(cmd) + "\n").encode("utf-8"))
            fhs[i].flush()
            harness.assert_port_free(harness.HOST, args.port0 if i == 0 else args.port1, f"node{i}")
            procs[i] = subprocess.Popen(cmd, stdout=fhs[i], stderr=subprocess.STDOUT, start_new_session=True)

        bases = [f"http://{harness.HOST}:{args.port0}", f"http://{harness.HOST}:{args.port1}"]
        health = [
            harness.wait_ready(bases[0], procs[0], logs[0]),
            harness.wait_ready(bases[1], procs[1], logs[1]),
        ]

        prompt_tokens = harness.tokenize_count(bases[0], prompt)
        for i, proc in enumerate(procs):
            assert proc is not None
            samplers[i] = harness.MemorySampler(proc)
            samplers[i].start()
            (out / f"node{i}-numastat-after-load.txt").write_text(numastat(proc.pid), encoding="utf-8")

        barrier = threading.Barrier(3)
        results: list[dict[str, Any] | None] = [None, None]
        errors: list[str | None] = [None, None]

        def worker(i: int) -> None:
            try:
                barrier.wait()
                t0 = time.time()
                response = harness.stream_completion(bases[i], request, visible_prefix="")
                elapsed = time.time() - t0
                results[i] = {
                    "response": response,
                    "summary": strict_summary(response, elapsed, prompt_tokens),
                }
            except Exception as exc:
                errors[i] = str(exc)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(2)]
        for thread in threads:
            thread.start()
        wall0 = time.time()
        barrier.wait()
        for thread in threads:
            thread.join()
        total_wall = time.time() - wall0

        for sampler in samplers:
            if sampler is not None:
                sampler.stop()
        for i, proc in enumerate(procs):
            assert proc is not None
            (out / f"node{i}-numastat-after-run.txt").write_text(numastat(proc.pid), encoding="utf-8")

        node_summaries = []
        for i, result in enumerate(results):
            if result is None:
                node_summaries.append({"all_ok": False, "error": errors[i]})
            else:
                summary = result["summary"]
                summary["peak_rss_mib_process_group_sum"] = None if samplers[i] is None else samplers[i].peak_rss_mib
                node_summaries.append(summary)
                jdump(out / f"node{i}-response.json", result["response"])

        strict_tokens_total = sum(int(s.get("strict_region_tokens") or 0) for s in node_summaries)
        prompt_tokens_total = prompt_tokens * 2
        max_first_content = max(float(s.get("time_to_first_content_seconds") or 0.0) for s in node_summaries)
        max_decode_seconds = max(float(s.get("decode_seconds_after_first_content") or 0.0) for s in node_summaries)
        summary = {
            "all_ok": all(bool(s.get("all_ok")) for s in node_summaries),
            "health": health,
            "model": args.model,
            "ctx_size": args.ctx_size,
            "flash_attn": args.flash_attn,
            "no_mmap": args.no_mmap,
            "total_wall_seconds": total_wall,
            "aggregate_prompt_tps_to_all_first_tokens": None if max_first_content <= 0.0 else prompt_tokens_total / max_first_content,
            "aggregate_strict_tokens_per_total_wall": strict_tokens_total / total_wall,
            "aggregate_strict_decode_tps_after_first_content": None if max_decode_seconds <= 0.0 else strict_tokens_total / max_decode_seconds,
            "nodes": node_summaries,
        }
        jdump(out / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["all_ok"] else 1
    finally:
        for sampler in samplers:
            if sampler is not None:
                sampler.stop()
        for proc in procs:
            kill_proc(proc)
        for fh in fhs:
            fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
