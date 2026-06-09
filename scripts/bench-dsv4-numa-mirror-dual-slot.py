#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from engine_test_harness import (  # noqa: E402
    EXPECTED_GENERATED,
    HOST,
    REQUEST_TIMEOUT,
    STOP_SENTINEL,
    append_dedup_rows,
    assert_port_free,
    assign_logical_spans,
    expected_checks,
    jdump,
    make_prompt,
    make_request,
    normalize_probability_list,
    parse_response_rows,
    resolve_server_bin,
    strict_range_for_expected,
    wait_port_closed,
    wait_ready,
)


DEFAULT_MODEL = "/home/mheiss/.cache/huggingface/hub/models--teamblobfish--DeepSeek-V4-Flash-GGUF/snapshots/b281094221a72c210a2b986709510b4c4b51b67e/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf"


def quiet_stream_completion(base_url: str, request: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(request).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    chunks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    content_parts: list[str] = []
    first_content_time: float | None = None
    start = time.time()

    try:
        opener = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"POST /completion failed with HTTP {exc.code} {exc.reason}. First 2000 chars:\n{body[:2000]}") from exc

    with opener as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunks.append(obj)
            s = str(obj.get("content", ""))
            if s:
                if first_content_time is None:
                    first_content_time = time.time()
                content_parts.append(s)
            append_dedup_rows(rows, normalize_probability_list(obj.get("completion_probabilities")))

    assign_logical_spans(rows)
    return {
        "content": "".join(content_parts),
        "completion_probabilities": [{"probs": rows}],
        "stream_chunks": chunks,
        "time_to_first_content_seconds": None if first_content_time is None else first_content_time - start,
        "elapsed_seconds": time.time() - start,
    }


def final_chunk(response: dict[str, Any]) -> dict[str, Any]:
    chunks = response.get("stream_chunks") or []
    for chunk in reversed(chunks):
        if isinstance(chunk, dict) and ("timings" in chunk or "tokens_predicted" in chunk):
            return chunk
    return {}


def summarize_response(name: str, response: dict[str, Any]) -> dict[str, Any]:
    rows = parse_response_rows(response)
    lo, hi, _ = strict_range_for_expected(rows, EXPECTED_GENERATED)
    strict_text = "" if lo is None or hi is None else "".join(str(r.get("chosen_token", "")) for r in rows[lo:hi + 1])
    check = expected_checks(response.get("content", ""))
    chunk = final_chunk(response)
    timings = chunk.get("timings") if isinstance(chunk.get("timings"), dict) else {}
    tokens_predicted = int(chunk.get("tokens_predicted") or timings.get("predicted_n") or len(rows))
    tokens_evaluated = int(chunk.get("tokens_evaluated") or timings.get("prompt_n") or 0)
    return {
        "name": name,
        "ok": bool(check["ok"] and strict_text == EXPECTED_GENERATED),
        "expected_text_ok": bool(check["ok"]),
        "strict_text_ok": strict_text == EXPECTED_GENERATED,
        "rows": len(rows),
        "strict_first": lo,
        "strict_last": hi,
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": tokens_evaluated,
        "elapsed_seconds": response.get("elapsed_seconds"),
        "time_to_first_content_seconds": response.get("time_to_first_content_seconds"),
        "server_timings": timings,
        "content": response.get("content", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two concurrent deterministic DeepSeek V4 requests against one NUMA mirror server.")
    parser.add_argument("build_dir", nargs="?", default="build-cpu-opt", help="build directory containing bin/llama-server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server-bin", default=None)
    parser.add_argument("--prefix", default="numactl --membind=0")
    parser.add_argument("--port", type=int, default=43182)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=52)
    parser.add_argument("--threads-batch", type=int, default=52)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--n-predict", type=int, default=192)
    parser.add_argument("--flash-attn", choices=["on", "off"], default="on")
    parser.add_argument("--numa", default="mirror")
    parser.set_defaults(no_mmap=True)
    parser.add_argument("--no-mmap", dest="no_mmap", action="store_true")
    parser.add_argument("--mmap", dest="no_mmap", action="store_false")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    server_bin = args.server_bin or str(Path(args.build_dir) / "bin" / "llama-server")
    server_bin = resolve_server_bin(server_bin)
    out_dir = Path(args.out_dir or f"dsv4-cascade-lake-results/numa-mirror-dual-slot-{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    base_url = f"http://{HOST}:{args.port}"

    cmd = [
        *shlex.split(args.prefix),
        server_bin,
        "-m", args.model,
        "--ctx-size", str(args.ctx_size),
        "--flash-attn", args.flash_attn,
        "--numa", args.numa,
        "--no-warmup",
        "--host", HOST,
        "--port", str(args.port),
        "-t", str(args.threads),
        "-tb", str(args.threads_batch),
        "-np", str(args.parallel),
        "--timeout", "1200",
    ]
    if args.no_mmap:
        cmd.append("--no-mmap")

    prompt = make_prompt(0, "deepseek")
    request = make_request(prompt)
    request["n_predict"] = args.n_predict

    assert_port_free(HOST, args.port, "numa_mirror_dual_slot")
    (out_dir / "server_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (out_dir / "request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")

    proc: subprocess.Popen[Any] | None = None
    with log_path.open("wb") as fh:
        try:
            print("Starting server:", " ".join(cmd))
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            health = wait_ready(base_url, proc, log_path, timeout=1200.0)
            print("Server health:", health)

            start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [
                    executor.submit(quiet_stream_completion, base_url, dict(request))
                    for _ in range(args.parallel)
                ]
                responses = [future.result() for future in futures]
            wall = time.time() - start
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
            wait_port_closed(HOST, args.port, timeout=15.0)

    summaries = [summarize_response(f"request_{i}", response) for i, response in enumerate(responses)]
    total_predicted = sum(item["tokens_predicted"] for item in summaries)
    result = {
        "ok": all(item["ok"] for item in summaries),
        "parallel": args.parallel,
        "wall_seconds": wall,
        "total_predicted_tokens": total_predicted,
        "aggregate_predicted_tps_wall": total_predicted / wall if wall > 0 else None,
        "summaries": summaries,
        "server_log": str(log_path),
    }
    jdump(out_dir / "summary.json", result)

    print("\n================ NUMA MIRROR DUAL SLOT ================")
    print("ok:", result["ok"])
    print("wall_seconds:", round(wall, 3))
    print("total_predicted_tokens:", total_predicted)
    print("aggregate_predicted_tps_wall:", result["aggregate_predicted_tps_wall"])
    for item in summaries:
        timings = item.get("server_timings") or {}
        print(
            item["name"],
            "ok=", item["ok"],
            "predicted=", item["tokens_predicted"],
            "elapsed=", round(float(item["elapsed_seconds"] or 0), 3),
            "server_pred_tps=", timings.get("predicted_per_second"),
        )
    print("summary:", out_dir / "summary.json")
    print("server log:", log_path)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
