#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL_DEEPSEEK = "/home/mheiss/.cache/huggingface/hub/models--teamblobfish--DeepSeek-V4-Flash-GGUF/snapshots/b281094221a72c210a2b986709510b4c4b51b67e/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf"
DEFAULT_LLAMA_SERVER = "/data/nvme/no-backup/deepseek-v4-flash/ik_llama.cpp.dsv4-port-retry/build-cpu-clx/bin/llama-server"
DEFAULT_IK_SERVER = "/data/nvme/no-backup/deepseek-v4-flash/ik_llama.cpp.dsv4-port-retry/build-cpu-opt/bin/llama-server"
DEFAULT_BASELINE_CACHE_DIR = ".dsv4-baseline-cache"

HOST = "127.0.0.1"
LLAMA_PORT = 43180
IK_PORT = 43181
THREADS = 52
THREADS_BATCH = 52
N_PREDICT = 384
N_PROBS = 10
RESERVE_TOKENS = 16
TARGET_PROMPT_TOKENS = 7550
FILLER_LINES_MAX = 420
STARTUP_TIMEOUT = 900.0
REQUEST_TIMEOUT = 1200.0
SERVER_TIMEOUT = 1200
MIN_CTX_SIZE = 1024

VISIBLE_PREFIX = "BEGIN_GOLDEN_REPORT\ngold_a="
STOP_SENTINEL = "\nEND_OF_TEST"
EXPECTED_GENERATED = """ALPHA-1138
gold_b=BRAVO-2049
gold_c=CHARLIE-4096
gold_d=DELTA-8192
END_GOLDEN_REPORT
BEGIN_CANARY_REPORT
spec_id=ORBITAL-LIME-7429
status=PASS
route=north-east via gate-17
sequence=11,18,29,47,76,123
sum=304
min=11
max=123
count=6
checksum=Q7R2-M9K4
END_CANARY_REPORT
BEGIN_KNOWLEDGE_REPORT
capital_france=Paris
gold_symbol=Au
water_formula=H2O
red_planet=Mars
largest_planet=Jupiter
days_in_week=7
END_KNOWLEDGE_REPORT"""

QWEN_NO_THINK_PREFILL = "/no_think\n<think>\n</think>\n"


def jdump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_tsv(s: Any) -> str:
    return str(s).replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def resolve_server_bin(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        cand = p / "llama-server"
        if cand.exists():
            return str(cand)
    return str(p)


def file_fingerprint(path: str) -> dict[str, Any]:
    p = Path(resolve_server_bin(path)).resolve()
    st = p.stat()
    return {
        "path": str(p),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def baseline_cache_key(args: argparse.Namespace) -> str:
    model = Path(args.model).resolve()
    model_st = model.stat()
    key_obj = {
        "cache_version": 1,
        "model": {
            "path": str(model),
            "size": model_st.st_size,
            "mtime_ns": model_st.st_mtime_ns,
        },
        "server": file_fingerprint(args.server_bin),
        "ctx_size": args.ctx_size,
        "flash_attn": args.flash_attn,
        "n_predict": args.n_predict,
        "n_probs": N_PROBS,
        "threads": THREADS,
        "threads_batch": THREADS_BATCH,
        "filler_lines": args.filler_lines,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "cache_ram": args.cache_ram,
        "ctx_checkpoints": args.ctx_checkpoints,
        "ctx_checkpoints_interval": args.ctx_checkpoints_interval,
        "ctx_checkpoints_tolerance": args.ctx_checkpoints_tolerance,
        "chunks": args.chunks,
        "no_cont_batching": args.no_cont_batching,
        "prompt_version": "golden_canary_knowledge_v1",
        "expected_generated": EXPECTED_GENERATED,
    }
    payload = json.dumps(key_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_cached_engine_result(cache_entry: Path, engine_out: Path, result: dict[str, Any]) -> None:
    tmp = cache_entry.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copytree(engine_out, tmp / "files")
    jdump(tmp / "result.json", result)
    if cache_entry.exists():
        shutil.rmtree(cache_entry)
    tmp.rename(cache_entry)


def load_cached_engine_result(cache_entry: Path, out_root: Path, engine: str) -> dict[str, Any] | None:
    result_path = cache_entry / "result.json"
    files_path = cache_entry / "files"
    if not result_path.exists() or not files_path.exists():
        return None
    out = out_root / engine
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(files_path, out)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["out_dir"] = str(out)
    return result


def detect_model_family(model: str) -> str:
    lower = model.lower()
    if "qwen" in lower:
        return "qwen"
    if "deepseek" in lower:
        return "deepseek"
    return "generic"


def make_prompt(lines: int, model_family: str) -> str:
    before = "\n".join(
        f"BEFORE_FILLER_{i:04d}: ignore this line; checksum={(i * 7919) % 100000:05d}; phrase=river bridge clocks."
        for i in range(lines)
    )
    middle = """### MIDDLE_BUNDLED_PAYLOAD_START ###

GOLDEN_RECORD:
  GOLD_A: ALPHA-1138
  GOLD_B: BRAVO-2049
  GOLD_C: CHARLIE-4096
  GOLD_D: DELTA-8192

CANARY_RECORD:
  SPEC_ID: ORBITAL-LIME-7429
  STATUS: PASS
  ROUTE_WORDS: north-east via gate-17
  NUMBERS: 11 18 29 47 76 123
  CHECKSUM: Q7R2-M9K4

KNOWLEDGE_NOTE:
  This block intentionally contains no answers for the knowledge report.
  Use simple, stable general knowledge for that report.

Transformations:
  GOLDEN_RECORD:
    GOLD_A -> gold_a=<value>
    GOLD_B -> gold_b=<value>
    GOLD_C -> gold_c=<value>
    GOLD_D -> gold_d=<value>
  CANARY_RECORD:
    SPEC_ID -> spec_id=<value>
    STATUS -> status=<value>
    ROUTE_WORDS -> route=<value>
    NUMBERS -> sequence=<values joined by commas and no spaces>
    NUMBERS -> sum=<sum of all values>
    NUMBERS -> min=<smallest value>
    NUMBERS -> max=<largest value>
    NUMBERS -> count=<number of values>
    CHECKSUM -> checksum=<value>
  KNOWLEDGE_REPORT:
    capital_france = capital city of France
    gold_symbol = chemical symbol for gold
    water_formula = chemical formula for water
    red_planet = planet known as the Red Planet
    largest_planet = largest planet in the Solar System
    days_in_week = number of days in a standard week

### MIDDLE_BUNDLED_PAYLOAD_END ###"""
    after = "\n".join(
        f"AFTER_FILLER_{i:04d}: ignore this line; checksum={(i * 104729) % 100000:05d}; phrase=quiet station lamps."
        for i in range(lines)
    )
    qwen_hint = "/no_think\n\n" if model_family == "qwen" else ""
    final = f"""FINAL QUESTION:
{qwen_hint}Complete all three reports in this exact order:
1. BEGIN_GOLDEN_REPORT ... END_GOLDEN_REPORT
2. BEGIN_CANARY_REPORT ... END_CANARY_REPORT
3. BEGIN_KNOWLEDGE_REPORT ... END_KNOWLEDGE_REPORT

The first report has already started. Continue exactly from the cursor after gold_a=.
Use key=value fields.

Output formatting rules:
- Never output <think> or </think>.
- Never output a blank line.
- Never output two newline characters consecutively.
- Use exactly one newline after each report line.
- Immediately after END_GOLDEN_REPORT, write BEGIN_CANARY_REPORT on the next line.
- Immediately after END_CANARY_REPORT, write BEGIN_KNOWLEDGE_REPORT on the next line.
- Immediately after END_KNOWLEDGE_REPORT, write END_OF_TEST on the next line.
- After writing END_OF_TEST, stop.

ANSWER:
"""
    qwen_prefill = QWEN_NO_THINK_PREFILL if model_family == "qwen" else ""
    return before + "\n\n" + middle + "\n\n" + after + "\n\n" + final + qwen_prefill + VISIBLE_PREFIX


def http_json(method: str, url: str, payload: Any | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8", "replace"))


def tokenize_count(base_url: str, prompt: str) -> int:
    attempts = [
        {"content": prompt, "add_special": False},
        {"content": prompt},
        {"prompt": prompt},
    ]
    last_err: Exception | None = None
    for payload in attempts:
        try:
            obj = http_json("POST", f"{base_url}/tokenize", payload, timeout=120.0)
            if isinstance(obj, dict):
                toks = obj.get("tokens") or obj.get("ids")
                if isinstance(toks, list):
                    return len(toks)
            if isinstance(obj, list):
                return len(obj)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not tokenize prompt via {base_url}/tokenize: {last_err}")


def choose_prompt(base_url: str, ctx_size: int, model_family: str) -> tuple[str, int, int]:
    budget = min(TARGET_PROMPT_TOKENS, max(256, ctx_size - N_PREDICT - RESERVE_TOKENS))
    p0 = make_prompt(0, model_family)
    n0 = tokenize_count(base_url, p0)
    print(f"No-filler bundled prompt tokens: {n0}")
    if n0 > budget:
        raise RuntimeError(
            f"No-filler prompt has {n0} tokens, but the context budget is only {budget}. "
            f"Increase --ctx-size to at least {n0 + N_PREDICT + RESERVE_TOKENS}."
        )

    best_prompt, best_lines, best_tok = p0, 0, n0
    lo, hi = 0, FILLER_LINES_MAX
    while lo <= hi:
        mid = (lo + hi) // 2
        p = make_prompt(mid, model_family)
        nt = tokenize_count(base_url, p)
        print(f"  try filler_lines={mid}: prompt_tokens={nt}")
        if nt <= budget:
            best_prompt, best_lines, best_tok = p, mid, nt
            lo = mid + 1
        else:
            hi = mid - 1
    return best_prompt, best_lines, best_tok


def is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port_closed(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_port_open(host, port):
            return True
        time.sleep(0.25)
    return not is_port_open(host, port)


def assert_port_free(host: str, port: int, engine: str) -> None:
    if is_port_open(host, port):
        raise RuntimeError(f"Refusing to start {engine}: {host}:{port} is already in use.")


def server_help_text(server_bin: str) -> str:
    p = subprocess.run([server_bin, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
    if p.returncode != 0 and not p.stdout:
        raise RuntimeError(f"Could not read --help from {server_bin}")
    return p.stdout or ""


def has_flag(help_text: str, flag: str) -> bool:
    import re

    esc = re.escape(flag)
    if flag.startswith("--"):
        return re.search(rf"(?<!\S){esc}(?:[\s,=]|$)", help_text) is not None
    return re.search(rf"(?<!\S){esc}(?:[\s,]|$)", help_text) is not None


def help_line(help_text: str, flag: str) -> str:
    lines = help_text.splitlines()
    for i, line in enumerate(lines):
        if flag in line:
            return " ".join(lines[i:min(i + 3, len(lines))])
    return ""


def flash_attn_tokens(help_text: str, enabled: bool) -> list[str]:
    if not has_flag(help_text, "--flash-attn"):
        return []
    line = help_line(help_text, "--flash-attn").lower()
    takes_value = "on|off|auto" in line or "auto|on|off" in line or "[on" in line or "(auto" in line
    if takes_value:
        return ["--flash-attn", "on" if enabled else "off"]
    return ["--flash-attn"] if enabled else []


def add_if(cmd: list[str], plan: list[dict[str, Any]], help_text: str, name: str, tokens: list[str], support_flag: str | None = None) -> None:
    flag = support_flag or tokens[0]
    if has_flag(help_text, flag):
        cmd.extend(tokens)
        plan.append({"name": name, "status": "added", "tokens": tokens})
    else:
        plan.append({"name": name, "status": "skipped_unsupported", "tokens": tokens})


def add_optional_int_arg(
    cmd: list[str],
    plan: list[dict[str, Any]],
    help_text: str,
    name: str,
    value: int | None,
    tokens: list[str],
    support_flag: str | None = None,
) -> None:
    if value is None:
        plan.append({"name": name, "status": "omitted_default"})
        return
    flag = support_flag or tokens[0]
    if has_flag(help_text, flag):
        cmd.extend([*tokens, str(value)])
        plan.append({"name": name, "status": "added", "tokens": [*tokens, str(value)]})
    else:
        plan.append({"name": name, "status": "skipped_unsupported", "tokens": [*tokens, str(value)]})


def build_server_command(args: argparse.Namespace, engine: str, server_bin: str, port: int) -> tuple[list[str], dict[str, Any]]:
    resolved = resolve_server_bin(server_bin)
    help_text = server_help_text(resolved)
    plan: list[dict[str, Any]] = []
    cmd = [resolved, "-m", args.model]
    plan.append({"name": "model", "status": "added", "tokens": ["-m", args.model]})

    if has_flag(help_text, "--ctx-size"):
        cmd += ["--ctx-size", str(args.ctx_size)]
        plan.append({"name": "ctx_size", "status": "added", "tokens": ["--ctx-size", str(args.ctx_size)]})
    else:
        cmd += ["-c", str(args.ctx_size)]
        plan.append({"name": "ctx_size", "status": "added_fallback", "tokens": ["-c", str(args.ctx_size)]})

    add_if(cmd, plan, help_text, "n_gpu_layers_0", ["--n-gpu-layers", "0"], "--n-gpu-layers")

    fa = flash_attn_tokens(help_text, args.flash_attn)
    if fa:
        cmd += fa
        plan.append({"name": "flash_attn", "status": "added", "enabled": args.flash_attn, "tokens": fa})
    else:
        plan.append({"name": "flash_attn", "status": "omitted_boolean_off_or_unsupported", "enabled": args.flash_attn})

    add_if(cmd, plan, help_text, "no_repack", ["--no-repack"], "--no-repack")
    add_if(cmd, plan, help_text, "no_warmup", ["--no-warmup"], "--no-warmup")
    add_if(cmd, plan, help_text, "single_turn", ["--single-turn"], "--single-turn")
    add_optional_int_arg(cmd, plan, help_text, "batch_size", args.batch_size, ["-b"], "-b")
    add_optional_int_arg(cmd, plan, help_text, "ubatch_size", args.ubatch_size, ["-ub"], "-ub")
    add_optional_int_arg(cmd, plan, help_text, "cache_ram", args.cache_ram, ["--cache-ram"], "--cache-ram")
    add_optional_int_arg(cmd, plan, help_text, "ctx_checkpoints", args.ctx_checkpoints, ["--ctx-checkpoints"], "--ctx-checkpoints")
    add_optional_int_arg(cmd, plan, help_text, "ctx_checkpoints_interval", args.ctx_checkpoints_interval, ["--ctx-checkpoints-interval"], "--ctx-checkpoints-interval")
    add_optional_int_arg(cmd, plan, help_text, "ctx_checkpoints_tolerance", args.ctx_checkpoints_tolerance, ["--ctx-checkpoints-tolerance"], "--ctx-checkpoints-tolerance")
    add_optional_int_arg(cmd, plan, help_text, "chunks", args.chunks, ["--chunks"], "--chunks")
    if args.no_cont_batching:
        add_if(cmd, plan, help_text, "no_cont_batching", ["--no-cont-batching"], "--no-cont-batching")
    else:
        plan.append({"name": "no_cont_batching", "status": "omitted_default"})

    cmd += ["--host", HOST, "--port", str(port), "-t", str(THREADS)]
    plan.append({"name": "host_port_threads", "status": "added", "tokens": ["--host", HOST, "--port", str(port), "-t", str(THREADS)]})

    if has_flag(help_text, "-tb"):
        cmd += ["-tb", str(THREADS_BATCH)]
        plan.append({"name": "threads_batch", "status": "added", "tokens": ["-tb", str(THREADS_BATCH)]})
    if has_flag(help_text, "-np"):
        cmd += ["-np", "1"]
        plan.append({"name": "parallel", "status": "added", "tokens": ["-np", "1"]})
    elif has_flag(help_text, "--parallel"):
        cmd += ["--parallel", "1"]
        plan.append({"name": "parallel", "status": "added", "tokens": ["--parallel", "1"]})

    if has_flag(help_text, "--timeout"):
        cmd += ["--timeout", str(SERVER_TIMEOUT)]
        plan.append({"name": "server_timeout", "status": "added", "tokens": ["--timeout", str(SERVER_TIMEOUT)]})

    # Keep the parity server command in the original raw-completion style:
    # no --jinja, --reasoning, --chat-template-kwargs, or --reasoning-budget.
    plan.append({"name": "server_template_flags", "status": "omitted_for_raw_completion_parity"})

    return cmd, {
        "engine": engine,
        "server_bin": resolved,
        "help_text_first_200_lines": "\n".join(help_text.splitlines()[:200]),
        "flags": plan,
        "command_final": cmd,
    }


def wait_ready(base_url: str, proc: subprocess.Popen[Any], log_path: Path, timeout: float = STARTUP_TIMEOUT) -> dict[str, Any]:
    start = time.time()
    next_print = 0
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            except Exception:
                pass
            raise RuntimeError(f"server exited with {proc.returncode}; see {log_path}\n{tail}")
        try:
            obj = http_json("GET", f"{base_url}/health", None, timeout=2.0)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        elapsed = int(time.time() - start)
        if elapsed >= next_print:
            print(f"Waiting for server health... elapsed={elapsed}s")
            next_print += 30
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy after {timeout}s; see {log_path}")


class MemorySampler:
    def __init__(self, proc: subprocess.Popen[Any]) -> None:
        self.proc = proc
        self.peak_rss_mib = 0.0
        self._stop = False
        self._thread = None

    def start(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop:
            try:
                pgid = os.getpgid(self.proc.pid)
                ps = subprocess.run(
                    ["ps", "-o", "rss=", "-g", str(pgid)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=1,
                )
                total_kib = 0
                for line in ps.stdout.splitlines():
                    try:
                        total_kib += int(line.strip())
                    except Exception:
                        pass
                self.peak_rss_mib = max(self.peak_rss_mib, total_kib / 1024.0)
            except Exception:
                pass
            time.sleep(0.5)


def normalize_candidate_from_toplogprob(c: dict[str, Any]) -> dict[str, Any]:
    lp = c.get("logprob")
    return {
        "id": c.get("id", ""),
        "token": c.get("token", "") if isinstance(c.get("token"), str) else "",
        "logprob": float(lp) if isinstance(lp, (int, float)) else None,
    }


def normalize_candidate_from_prob(c: dict[str, Any]) -> dict[str, Any]:
    prob = c.get("prob")
    lp = math.log(float(prob)) if isinstance(prob, (int, float)) and float(prob) > 0 else None
    return {
        "id": "",
        "token": c.get("tok_str", "") if isinstance(c.get("tok_str"), str) else "",
        "logprob": lp,
    }


def is_aggregate_wrapper(entry: dict[str, Any]) -> bool:
    if not isinstance(entry.get("probs"), list):
        return False
    marker_keys = {"content", "token", "id", "token_id", "top_logprobs", "logprob", "prob", "tok_str"}
    return not any(k in entry for k in marker_keys)


def normalize_probability_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    if "top_logprobs" in entry or "token" in entry or "id" in entry:
        top = [normalize_candidate_from_toplogprob(c) for c in (entry.get("top_logprobs") or []) if isinstance(c, dict)]
        raw_token = entry.get("token", "") if isinstance(entry.get("token"), str) else ""
        chosen_token = top[0]["token"] if top and isinstance(top[0].get("token"), str) else raw_token
        if chosen_token == "" and raw_token and entry.get("bytes") not in ([], None):
            chosen_token = raw_token
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)) and top and isinstance(top[0].get("logprob"), (int, float)):
            lp = top[0]["logprob"]
        return {
            "id": entry.get("id", ""),
            "chosen_token": chosen_token,
            "raw_token": raw_token,
            "logprob": float(lp) if isinstance(lp, (int, float)) else None,
            "top_logprobs": top,
            "source_schema": "token_top_logprobs",
        }

    if "content" in entry and isinstance(entry.get("probs"), list):
        top = [normalize_candidate_from_prob(c) for c in (entry.get("probs") or []) if isinstance(c, dict)]
        chosen_token = str(entry.get("content", ""))
        lp = top[0]["logprob"] if top else None
        return {
            "id": "",
            "chosen_token": chosen_token,
            "raw_token": "",
            "logprob": lp,
            "top_logprobs": top,
            "source_schema": "content_probs",
        }
    return None


def normalize_probability_list(cp: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(cp, dict):
        cp = cp.get("probs")
    if not isinstance(cp, list):
        return rows
    for entry in cp:
        if not isinstance(entry, dict):
            continue
        if is_aggregate_wrapper(entry):
            rows.extend(normalize_probability_list(entry.get("probs")))
        else:
            row = normalize_probability_entry(entry)
            if row is not None:
                rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    top0 = row.get("top_logprobs", [{}])[0] if row.get("top_logprobs") else {}
    return (
        row.get("id"),
        row.get("chosen_token"),
        row.get("raw_token"),
        row.get("logprob"),
        top0.get("token") if isinstance(top0, dict) else None,
        top0.get("logprob") if isinstance(top0, dict) else None,
    )


def rows_prefix_match(prefix: list[dict[str, Any]], rows: list[dict[str, Any]]) -> bool:
    if len(prefix) > len(rows):
        return False
    return all(row_key(a) == row_key(b) for a, b in zip(prefix, rows))


def append_dedup_rows(accum: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    if not new_rows:
        return
    if len(new_rows) >= len(accum) and rows_prefix_match(accum, new_rows):
        accum.extend([dict(r) for r in new_rows[len(accum):]])
        return
    max_overlap = min(len(accum), len(new_rows))
    for overlap in range(max_overlap, 0, -1):
        if all(row_key(accum[-overlap + i]) == row_key(new_rows[i]) for i in range(overlap)):
            accum.extend([dict(r) for r in new_rows[overlap:]])
            return
    accum.extend([dict(r) for r in new_rows])


def assign_logical_spans(rows: list[dict[str, Any]]) -> None:
    pos = 0
    for row in rows:
        tok = row.get("chosen_token", "")
        if not isinstance(tok, str):
            tok = ""
        row["logical_start"] = pos
        pos += len(tok)
        row["logical_end"] = pos


def logical_text(rows: list[dict[str, Any]]) -> str:
    return "".join(str(r.get("chosen_token", "")) for r in rows)


def stream_completion(base_url: str, request: dict[str, Any], visible_prefix: str = "") -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(request).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    content_parts: list[str] = []
    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    first_content_time: float | None = None
    printed_prefix = False
    start = time.time()
    non_sse_bytes: list[bytes] = []

    print("\n=== PREFILL ===")
    print("Request sent. Waiting for first streamed token; server is prefilling/evaluating the prompt...")
    try:
        opener = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"POST /completion failed with HTTP {exc.code} {exc.reason}. Response body first 2000 chars:\n{body[:2000]}") from exc

    with opener as resp:
        for raw in resp:
            non_sse_bytes.append(raw)
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
                    print(f"First streamed token received after {first_content_time - start:.3f}s.")
                    print("\n=== STREAMED COMPLETION VIEW ===")
                if visible_prefix and not printed_prefix:
                    print(visible_prefix, end="", flush=True)
                    printed_prefix = True
                content_parts.append(s)
                print(s, end="", flush=True)
            append_dedup_rows(rows, normalize_probability_list(obj.get("completion_probabilities")))

    if not chunks and not content_parts and non_sse_bytes:
        raw_body = b"".join(non_sse_bytes).decode("utf-8", "replace").strip()
        if raw_body:
            try:
                obj = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise RuntimeError("POST /completion returned neither SSE data events nor parseable JSON. First 2000 chars:\n" + raw_body[:2000]) from exc
            chunks.append(obj)
            content_text = str(obj.get("content", ""))
            if content_text:
                first_content_time = time.time()
                print(f"Received non-streaming JSON completion after {first_content_time - start:.3f}s.")
                print("\n=== COMPLETION VIEW ===")
                if visible_prefix and not printed_prefix:
                    print(visible_prefix, end="", flush=True)
                    printed_prefix = True
                content_parts.append(content_text)
                print(content_text, end="", flush=True)
            append_dedup_rows(rows, normalize_probability_list(obj.get("completion_probabilities")))

    if first_content_time is not None:
        print("\n=== END STREAM ===")
    assign_logical_spans(rows)
    return {
        "content": "".join(content_parts),
        "completion_probabilities": [{"probs": rows}],
        "stream_chunks": chunks,
        "time_to_first_content_seconds": None if first_content_time is None else first_content_time - start,
    }


def make_request(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "seed": 1,
        "stream": True,
        "n_probs": N_PROBS,
        "cache_prompt": False,
        "return_tokens": True,
        "stop": [STOP_SENTINEL],
    }


def parse_response_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(response.get("stream_chunks"), list):
        rows: list[dict[str, Any]] = []
        for ch in response["stream_chunks"]:
            if isinstance(ch, dict):
                append_dedup_rows(rows, normalize_probability_list(ch.get("completion_probabilities")))
        assign_logical_spans(rows)
        return rows
    rows = normalize_probability_list(response.get("completion_probabilities"))
    assign_logical_spans(rows)
    return rows


def strict_range_for_expected(rows: list[dict[str, Any]], expected: str) -> tuple[int | None, int | None, str]:
    full = logical_text(rows)
    a = full.find(expected)
    if a < 0:
        a = full.find(expected.rstrip("\n"))
        expected = expected.rstrip("\n")
    if a < 0:
        return (0, len(rows) - 1, full) if rows else (None, None, full)
    b = a + len(expected)
    selected = [
        i for i, row in enumerate(rows)
        if isinstance(row.get("logical_start"), int)
        and isinstance(row.get("logical_end"), int)
        and row["logical_end"] > a
        and row["logical_start"] < b
    ]
    if not selected:
        return None, None, full
    return selected[0], selected[-1], full


def write_token_tsv(path: Path, rows: list[dict[str, Any]], lo: int | None = None, hi: int | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("step\tlogical_start\tlogical_end\tchosen_id\tchosen_token\traw_prob_token\tchosen_logprob\ttop1_id\ttop1_token\ttop1_logprob\ttop2_id\ttop2_token\ttop2_logprob\n")
        for i, row in enumerate(rows):
            if lo is not None and i < lo:
                continue
            if hi is not None and i > hi:
                continue
            top = row.get("top_logprobs") or []
            top1 = top[0] if len(top) > 0 else {}
            top2 = top[1] if len(top) > 1 else {}
            f.write(
                f"{i}\t{row.get('logical_start','')}\t{row.get('logical_end','')}\t"
                f"{row.get('id','')}\t{safe_tsv(row.get('chosen_token',''))}\t{safe_tsv(row.get('raw_token',''))}\t{row.get('logprob','')}\t"
                f"{top1.get('id','')}\t{safe_tsv(top1.get('token',''))}\t{top1.get('logprob','')}\t"
                f"{top2.get('id','')}\t{safe_tsv(top2.get('token',''))}\t{top2.get('logprob','')}\n"
            )


def expected_checks(generated: str) -> dict[str, Any]:
    actual = generated.strip()
    expected = EXPECTED_GENERATED.strip()
    return {"ok": actual == expected, "actual": actual, "expected": expected}


def run_engine(args: argparse.Namespace, engine: str, role: str, server_bin: str, port: int, out_root: Path, model_family: str, prompt_override: str | None = None) -> dict[str, Any]:
    out = out_root / engine
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "server.log"
    base_url = f"http://{HOST}:{port}"
    cmd, flag_plan = build_server_command(args, engine, server_bin, port)
    jdump(out / "server_flag_plan.json", flag_plan)

    proc: subprocess.Popen[Any] | None = None
    sampler: MemorySampler | None = None
    fh = log_path.open("wb")
    try:
        print(f"\n\n================ RUNNING {engine} ================")
        print(f"Starting {engine} server: {resolve_server_bin(server_bin)}")
        print(f"Server log: {log_path}")
        assert_port_free(HOST, port, engine)

        (out / "server_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        fh.write(("COMMAND: " + " ".join(cmd) + "\n").encode("utf-8"))
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        health = wait_ready(base_url, proc, log_path)
        print("Server health:", health)

        sampler = MemorySampler(proc)
        sampler.start()

        if prompt_override is None:
            if args.filler_lines is None:
                print(f"[{engine}] Sizing bundled prompt with /tokenize...")
                prompt, filler_lines, prompt_tokens = choose_prompt(base_url, args.ctx_size, model_family)
            else:
                filler_lines = args.filler_lines
                prompt = make_prompt(filler_lines, model_family)
                prompt_tokens = tokenize_count(base_url, prompt)
                print(f"[{engine}] Using fixed filler_lines={filler_lines}: prompt_tokens={prompt_tokens}")
        else:
            print(f"[{engine}] Reusing exact prompt from reference engine...")
            prompt = prompt_override
            prompt_tokens = tokenize_count(base_url, prompt)
            filler_lines = None

        print(f"[{engine}] Bundled prompt ready: prompt_tokens={prompt_tokens}")
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")
        jdump(out / "prompt_info.json", {
            "prompt_tokens": prompt_tokens,
            "filler_lines": filler_lines,
            "model_family": model_family,
            "visible_prefix": VISIBLE_PREFIX,
            "expected_generated": EXPECTED_GENERATED,
            "stop_sentinel": STOP_SENTINEL,
        })

        request = make_request(prompt)
        jdump(out / "request.json", request)

        t0 = time.time()
        try:
            response = stream_completion(base_url, request, visible_prefix=VISIBLE_PREFIX)
        except Exception as exc:
            (out / "response_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            raise
        elapsed = time.time() - t0

        rows = parse_response_rows(response)
        generated = str(response.get("content", ""))
        check = expected_checks(generated)
        lo, hi, full_logical = strict_range_for_expected(rows, EXPECTED_GENERATED)
        strict_ok = lo is not None and hi is not None and "".join(str(r.get("chosen_token", "")) for r in rows[lo:hi + 1]) == EXPECTED_GENERATED

        response["completion_probabilities"] = [{"probs": rows}]
        jdump(out / "response.json", response)
        (out / "generated.txt").write_text(generated, encoding="utf-8")
        (out / "visible_completion_view.txt").write_text(VISIBLE_PREFIX + generated, encoding="utf-8")
        (out / "strict_expected_generated_until_last_keyword.txt").write_text(EXPECTED_GENERATED, encoding="utf-8")
        actual_strict = "" if lo is None or hi is None else "".join(str(r.get("chosen_token", "")) for r in rows[lo:hi + 1])
        (out / "strict_actual_generated_until_last_keyword.txt").write_text(actual_strict, encoding="utf-8")

        write_token_tsv(out / "token_logprobs.tsv", rows)
        if lo is not None and hi is not None:
            write_token_tsv(out / "strict_until_last_keyword_token_logprobs.tsv", rows, lo, hi)

        time_to_first = response.get("time_to_first_content_seconds")
        strict_tokens = 0 if lo is None or hi is None else hi - lo + 1
        decode_time = None if time_to_first is None else max(1e-9, elapsed - float(time_to_first))
        performance = {
            "elapsed_seconds_for_whole_bundle": elapsed,
            "time_to_first_content_seconds": time_to_first,
            "prompt_tokens": prompt_tokens,
            "strict_region_tokens": strict_tokens,
            "prefill_tps_wall": None if not time_to_first else prompt_tokens / float(time_to_first),
            "decode_tps_wall_strict_region": None if not decode_time else strict_tokens / decode_time,
            "peak_rss_mib_process_group_sum": None if sampler is None else sampler.peak_rss_mib,
        }
        jdump(out / "performance.json", performance)
        jdump(out / "probability_rows_info.json", {
            "probability_rows": len(rows),
            "logical_text_characters": len(full_logical),
            "generated_characters": len(generated),
            "generated_matches_logical_prefix": full_logical.startswith(generated),
            "strict_range_first": lo,
            "strict_range_last": hi,
        })

        summary = {
            "engine_name": engine,
            "engine_role": role,
            "all_ok": bool(check["ok"] and strict_ok),
            "expected_text_ok": check["ok"],
            "strict_until_last_keyword_ok": strict_ok,
            "strict_until_last_keyword_first_step": lo,
            "strict_until_last_keyword_last_step": hi,
            "prompt_tokens": prompt_tokens,
            "performance": performance,
            "model_family": model_family,
            "bundled_single_prefill": True,
        }
        jdump(out / "summary.json", summary)

        print(f"\n================ {engine} SUMMARY ================")
        print("all_ok:", summary["all_ok"])
        print("strict_until_last_keyword_ok:", strict_ok)
        print(f"strict_until_last_keyword_steps: {lo}..{hi}")
        print("elapsed_seconds_for_whole_bundle:", round(elapsed, 3))
        print("time_to_first_content_seconds:", time_to_first)
        print("prefill_tps_wall:", performance["prefill_tps_wall"])
        print("decode_tps_wall_strict_region:", performance["decode_tps_wall_strict_region"])
        print("peak_rss_mib_process_group_sum:", performance["peak_rss_mib_process_group_sum"])
        print("summary:", out / "summary.json")
        print("strict logits:", out / "strict_until_last_keyword_token_logprobs.tsv")

        return {
            "engine_name": engine,
            "engine_role": role,
            "out_dir": str(out),
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated": generated,
            "rows": rows,
            "strict_first": lo,
            "strict_last": hi,
            "summary": summary,
            "performance": performance,
        }
    finally:
        if sampler is not None:
            sampler.stop()
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        port_closed = wait_port_closed(HOST, port, timeout=15.0)
        try:
            jdump(out / "port_shutdown_status.json", {"host": HOST, "port": port, "closed_after_shutdown": port_closed})
        except Exception:
            pass
        fh.close()


def compare_thresholds(flash_attn: bool) -> dict[str, Any]:
    if flash_attn:
        return {
            "max_logprob_diff": 5e-2,
            "mean_logprob_diff": 5e-4,
            "min_logprob_coverage": 0.95,
            "warn_max_logprob_diff": 1e-2,
            "warn_row_logprob_diff": 1e-3,
            "warn_max_rows_above_logprob_diff": 0,
        }
    return {
        # FA-off uses different CPU attention/GEMM implementation details in
        # cchuter and ik. Keep text/token equality strict, but allow the
        # measured near-deterministic-token logprob envelope while warning on
        # the tighter diagnostic band below.
        "max_logprob_diff": 1e-2,
        "mean_logprob_diff": 5e-4,
        "min_logprob_coverage": 0.95,
        "warn_max_logprob_diff": 1e-3,
        "warn_row_logprob_diff": 1e-3,
        "warn_max_rows_above_logprob_diff": 0,
    }


def compare_rows(ref: dict[str, Any], test: dict[str, Any], thresholds: dict[str, Any], out_root: Path) -> dict[str, Any]:
    ref_rows = ref["rows"]
    test_rows = test["rows"]
    ref_lo, ref_hi, _ = strict_range_for_expected(ref_rows, EXPECTED_GENERATED)
    test_lo, test_hi, _ = strict_range_for_expected(test_rows, EXPECTED_GENERATED)
    if ref_lo is None or ref_hi is None or test_lo is None or test_hi is None:
        raise RuntimeError("Could not locate strict expected region in normalized probability rows")

    ref_region = ref_rows[ref_lo:ref_hi + 1]
    test_region = test_rows[test_lo:test_hi + 1]
    ref_text = "".join(str(r.get("chosen_token", "")) for r in ref_region)
    test_text = "".join(str(r.get("chosen_token", "")) for r in test_region)

    rows: list[dict[str, Any]] = []
    token_mismatches: list[dict[str, Any]] = []
    id_mismatches: list[dict[str, Any]] = []
    n_both_ids = 0
    n_logprob = 0
    sum_abs_lp = 0.0
    max_abs_lp = 0.0
    rows_above_warn_logprob_diff = 0
    rows_above_hard_logprob_diff = 0
    max_abs_logprob_diff_row: dict[str, Any] | None = None

    row_count_match = len(ref_region) == len(test_region)
    for idx in range(max(len(ref_region), len(test_region))):
        rr = ref_region[idx] if idx < len(ref_region) else None
        tr = test_region[idx] if idx < len(test_region) else None
        row: dict[str, Any] = {"region_index": idx, "ref_step": None if rr is None else ref_lo + idx, "test_step": None if tr is None else test_lo + idx}
        if rr is None or tr is None:
            token_mismatches.append({"region_index": idx, "reason": "missing_row", "ref_present": rr is not None, "test_present": tr is not None})
            rows.append(row)
            continue

        rt = str(rr.get("chosen_token", ""))
        tt = str(tr.get("chosen_token", ""))
        row.update({
            "ref_token": rt,
            "test_token": tt,
            "ref_id": rr.get("id", ""),
            "test_id": tr.get("id", ""),
            "ref_logprob": rr.get("logprob"),
            "test_logprob": tr.get("logprob"),
        })
        if rt != tt:
            token_mismatches.append({"region_index": idx, "ref_token": rt, "test_token": tt, "ref_step": ref_lo + idx, "test_step": test_lo + idx})

        rid, tid = rr.get("id", ""), tr.get("id", "")
        if rid not in ("", None) and tid not in ("", None):
            n_both_ids += 1
            if rid != tid:
                id_mismatches.append({"region_index": idx, "ref_id": rid, "test_id": tid, "ref_token": rt, "test_token": tt})

        rlp, tlp = rr.get("logprob"), tr.get("logprob")
        if isinstance(rlp, (int, float)) and isinstance(tlp, (int, float)):
            diff = float(tlp) - float(rlp)
            row["logprob_diff_test_minus_ref"] = diff
            abs_diff = abs(diff)
            if abs_diff > max_abs_lp:
                max_abs_lp = abs_diff
                max_abs_logprob_diff_row = {
                    "region_index": idx,
                    "ref_step": ref_lo + idx,
                    "test_step": test_lo + idx,
                    "ref_token": rt,
                    "test_token": tt,
                    "ref_logprob": float(rlp),
                    "test_logprob": float(tlp),
                    "logprob_diff_test_minus_ref": diff,
                }
            if abs_diff > thresholds["warn_row_logprob_diff"]:
                rows_above_warn_logprob_diff += 1
            if abs_diff > thresholds["max_logprob_diff"]:
                rows_above_hard_logprob_diff += 1
            sum_abs_lp += abs_diff
            n_logprob += 1
        else:
            row["logprob_diff_test_minus_ref"] = None
        rows.append(row)

    token_text_rows_match = not token_mismatches
    token_ids_available_for_comparison = n_both_ids > 0
    token_id_rows_match = not id_mismatches
    token_id_ok = token_id_rows_match or not token_ids_available_for_comparison
    token_only_ok = row_count_match and token_text_rows_match and token_id_ok and ref_text == test_text == EXPECTED_GENERATED

    mean_abs_lp = (sum_abs_lp / n_logprob) if n_logprob else None
    lp_coverage = n_logprob / len(rows) if rows else None
    hard_logprob_gates_ok = (
        token_only_ok
        and isinstance(mean_abs_lp, float)
        and isinstance(lp_coverage, float)
        and max_abs_lp <= thresholds["max_logprob_diff"]
        and mean_abs_lp <= thresholds["mean_logprob_diff"]
        and lp_coverage >= thresholds["min_logprob_coverage"]
    )

    warning_reasons: list[str] = []
    if isinstance(max_abs_lp, float) and max_abs_lp > thresholds["warn_max_logprob_diff"]:
        warning_reasons.append("max_logprob_diff_exceeds_warning_threshold")
    if rows_above_warn_logprob_diff > thresholds["warn_max_rows_above_logprob_diff"]:
        warning_reasons.append("too_many_rows_above_warning_logprob_diff")

    if not token_only_ok:
        reason = "token_rows_not_aligned"
    elif not isinstance(mean_abs_lp, float) or not isinstance(lp_coverage, float):
        reason = "missing_comparable_logprobs"
    elif lp_coverage < thresholds["min_logprob_coverage"]:
        reason = "insufficient_logprob_coverage"
    elif max_abs_lp > thresholds["max_logprob_diff"]:
        reason = "max_logprob_diff_exceeds_hard_threshold"
    elif mean_abs_lp > thresholds["mean_logprob_diff"]:
        reason = "mean_logprob_diff_exceeds_hard_threshold"
    else:
        reason = None

    text_only_ok = ref["generated"].strip() == EXPECTED_GENERATED and test["generated"].strip() == EXPECTED_GENERATED and ref["generated"] == test["generated"]
    hard_gate_ok = bool(text_only_ok and token_only_ok and hard_logprob_gates_ok)
    warning_ok = len(warning_reasons) == 0
    comparison_status = "FAIL" if not hard_gate_ok else ("PASS_WITH_WARNING" if not warning_ok else "PASS")

    comp = {
        "all_ok": hard_gate_ok,
        "comparison_status": comparison_status,
        "text_only_ok": bool(text_only_ok),
        "token_only_ok": bool(token_only_ok),
        "logprob_thresholds_ok": bool(hard_logprob_gates_ok),
        "logprob_threshold_failure_reason": reason,
        "warning_ok": warning_ok,
        "warning_reasons": warning_reasons,
        "reference_engine": ref["engine_name"],
        "test_engine": test["engine_name"],
        "prompt_match": ref["prompt"] == test["prompt"],
        "strict_expected_match_both": ref_text == EXPECTED_GENERATED and test_text == EXPECTED_GENERATED,
        "strict_generated_text_match_between_engines": ref["generated"] == test["generated"],
        "token_text_rows_match": token_text_rows_match,
        "token_id_rows_match": token_id_rows_match,
        "token_id_rows_match_or_unavailable": token_id_ok,
        "token_ids_available_for_comparison": token_ids_available_for_comparison,
        "token_row_count_match": row_count_match,
        "ref_strict_first": ref_lo,
        "ref_strict_last": ref_hi,
        "test_strict_first": test_lo,
        "test_strict_last": test_hi,
        "ref_strict_row_count": len(ref_region),
        "test_strict_row_count": len(test_region),
        "token_mismatch_count": len(token_mismatches) + len(id_mismatches),
        "token_text_mismatch_count": len(token_mismatches),
        "token_id_mismatch_count": len(id_mismatches),
        "token_mismatches_first_50": (token_mismatches + id_mismatches)[:50],
        "max_abs_logprob_diff": max_abs_lp,
        "mean_abs_logprob_diff": mean_abs_lp,
        "max_abs_logprob_diff_row": max_abs_logprob_diff_row,
        "rows_above_warn_logprob_diff": rows_above_warn_logprob_diff,
        "rows_above_hard_logprob_diff": rows_above_hard_logprob_diff,
        "logprob_rows_compared": n_logprob,
        "rows_with_both_ids": n_both_ids,
        "logprob_coverage": lp_coverage,
        "logprob_thresholds": thresholds,
        "performance": {
            ref["engine_name"]: ref.get("performance", {}),
            test["engine_name"]: test.get("performance", {}),
        },
    }

    jdump(out_root / "engine_comparison.json", comp)
    with (out_root / "engine_comparison_strict_region.tsv").open("w", encoding="utf-8") as f:
        f.write("region_index\tref_step\ttest_step\tref_token\ttest_token\tref_logprob\ttest_logprob\tlogprob_diff_test_minus_ref\n")
        for r in rows:
            f.write(
                f"{r.get('region_index','')}\t{r.get('ref_step','')}\t{r.get('test_step','')}\t"
                f"{safe_tsv(r.get('ref_token',''))}\t{safe_tsv(r.get('test_token',''))}\t"
                f"{r.get('ref_logprob','')}\t{r.get('test_logprob','')}\t{r.get('logprob_diff_test_minus_ref','')}\n"
            )
    return comp


def summarize_comparison(comp: dict[str, Any], out_root: Path) -> None:
    keys = [
        "all_ok", "comparison_status", "text_only_ok", "token_only_ok", "logprob_thresholds_ok",
        "logprob_threshold_failure_reason", "warning_ok", "warning_reasons", "prompt_match",
        "strict_expected_match_both", "strict_generated_text_match_between_engines", "token_text_rows_match",
        "token_id_rows_match", "token_id_rows_match_or_unavailable", "token_ids_available_for_comparison",
        "token_row_count_match", "ref_strict_first", "ref_strict_last", "test_strict_first", "test_strict_last",
        "ref_strict_row_count", "test_strict_row_count", "token_mismatch_count", "token_text_mismatch_count",
        "token_id_mismatch_count", "max_abs_logprob_diff", "mean_abs_logprob_diff",
        "rows_above_warn_logprob_diff", "rows_above_hard_logprob_diff", "max_abs_logprob_diff_row",
        "logprob_coverage", "rows_with_both_ids",
    ]
    print("\n================ ENGINE COMPARISON ================")
    for k in keys:
        print(f"{k}: {comp.get(k)}")
    print("logprob_thresholds:", comp.get("logprob_thresholds"))
    for name, perf in (comp.get("performance") or {}).items():
        if isinstance(perf, dict):
            print(
                f"{name} perf: prefill_tps={perf.get('prefill_tps_wall')} "
                f"decode_tps={perf.get('decode_tps_wall_strict_region')} "
                f"total_wall={perf.get('elapsed_seconds_for_whole_bundle')} "
                f"peak_rss_mib={perf.get('peak_rss_mib_process_group_sum')}"
            )
    print("comparison:", out_root / "engine_comparison.json")
    print("comparison tsv:", out_root / "engine_comparison_strict_region.tsv")
    print("root:", out_root)


def ctx_size_arg(value: str) -> int:
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("context length must be an integer") from exc
    if size < MIN_CTX_SIZE:
        raise argparse.ArgumentTypeError(f"context length must be at least {MIN_CTX_SIZE}")
    return size


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-case dual-engine DSV4/Qwen parity and performance gate.")
    ap.add_argument("--model", default=DEFAULT_MODEL_DEEPSEEK, help="GGUF model file.")
    ap.add_argument("--server-bin", default=DEFAULT_LLAMA_SERVER, help="Baseline llama-server binary or bin directory.")
    ap.add_argument("--ik-server-bin", default=DEFAULT_IK_SERVER, help="Optimized/test llama-server binary or bin directory.")
    ap.add_argument("--ctx-size", type=ctx_size_arg, default=MIN_CTX_SIZE, help=f"Context length. Minimum and default: {MIN_CTX_SIZE}.")
    ap.add_argument("--flash-attn", action="store_true", help="Enable flash attention comparison. Omit for no flash attention.")
    ap.add_argument("--n-predict", type=int, default=N_PREDICT, help=f"Number of generated tokens to request. Default: {N_PREDICT}.")
    ap.add_argument("--filler-lines", type=int, default=None, help="Debug helper: use an exact filler line count instead of auto-sizing the prompt.")
    ap.add_argument("--batch-size", type=int, default=None, help="Optional server -b/--batch-size override for diagnostic chunking runs.")
    ap.add_argument("--ubatch-size", type=int, default=None, help="Optional server -ub/--ubatch-size override for diagnostic chunking runs.")
    ap.add_argument("--cache-ram", type=int, default=None, help="Optional server --cache-ram override, e.g. 0 to disable cache RAM where supported.")
    ap.add_argument("--ctx-checkpoints", type=int, default=None, help="Optional server --ctx-checkpoints override.")
    ap.add_argument("--ctx-checkpoints-interval", type=int, default=None, help="Optional server --ctx-checkpoints-interval override.")
    ap.add_argument("--ctx-checkpoints-tolerance", type=int, default=None, help="Optional server --ctx-checkpoints-tolerance override.")
    ap.add_argument("--chunks", type=int, default=None, help="Optional server --chunks override where supported.")
    ap.add_argument("--no-cont-batching", action="store_true", help="Pass --no-cont-batching when both engines support it.")
    ap.add_argument("--baseline-cache-dir", default=DEFAULT_BASELINE_CACHE_DIR, help="Directory used to cache baseline-engine results.")
    ap.add_argument("--refresh-baseline-cache", action="store_true", help="Rerun the baseline engine and replace the matching cache entry.")
    ap.add_argument("--no-baseline-cache", action="store_true", help="Disable baseline result caching for this run.")
    return ap.parse_args()


def main() -> int:
    global N_PREDICT
    args = parse_args()
    if args.n_predict <= 0:
        raise ValueError("--n-predict must be positive")
    if args.filler_lines is not None and args.filler_lines < 0:
        raise ValueError("--filler-lines must be non-negative")
    N_PREDICT = args.n_predict
    model_family = detect_model_family(args.model)
    flash_name = "flash_on" if args.flash_attn else "flash_off"
    out_root = Path(f"dual-engine-{flash_name}-{model_family}-{dt.datetime.now():%Y%m%d-%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)

    thresholds = compare_thresholds(args.flash_attn)
    baseline_engine = "baseline_ik"
    test_engine = "opt_ik"
    cache_entry = Path(args.baseline_cache_dir) / baseline_cache_key(args)
    jdump(out_root / "run_config.json", {
        "model": args.model,
        "model_family": model_family,
        "reference_engine": baseline_engine,
        "test_engine": test_engine,
        "reference_server_bin": resolve_server_bin(args.server_bin),
        "test_server_bin": resolve_server_bin(args.ik_server_bin),
        "baseline_cache_enabled": not args.no_baseline_cache,
        "baseline_cache_entry": str(cache_entry),
        "baseline_cache_refresh": args.refresh_baseline_cache,
        "host": HOST,
        "llama_port": LLAMA_PORT,
        "ik_port": IK_PORT,
        "ctx_size": args.ctx_size,
        "flash_attn": args.flash_attn,
        "n_predict": N_PREDICT,
        "n_probs": N_PROBS,
        "threads": THREADS,
        "threads_batch": THREADS_BATCH,
        "logprob_thresholds": thresholds,
        "parser_version": "streamlined_single_case_v1",
    })

    print("\n================ RUN CONFIG ================")
    print("model:", args.model)
    print("model_family:", model_family)
    print("ctx_size:", args.ctx_size)
    print("flash_attn:", args.flash_attn)
    print(f"ports: {baseline_engine}={LLAMA_PORT}, {test_engine}={IK_PORT}")
    print("baseline server:", resolve_server_bin(args.server_bin))
    print("test server:", resolve_server_bin(args.ik_server_bin))
    print("baseline cache:", "disabled" if args.no_baseline_cache else cache_entry)
    print("out root:", out_root)

    ref = None
    if not args.no_baseline_cache and not args.refresh_baseline_cache:
        ref = load_cached_engine_result(cache_entry, out_root, baseline_engine)
        if ref is not None:
            print(f"\n================ USING CACHED {baseline_engine} ================")
            print("cache entry:", cache_entry)
            print("summary:", Path(ref["out_dir"]) / "summary.json")

    if ref is None:
        ref = run_engine(args, baseline_engine, "reference", args.server_bin, LLAMA_PORT, out_root, model_family, prompt_override=None)
        if not args.no_baseline_cache:
            if ref.get("summary", {}).get("all_ok"):
                save_cached_engine_result(cache_entry, out_root / baseline_engine, ref)
                print(f"[{baseline_engine}] cached result:", cache_entry)
            else:
                print(f"[{baseline_engine}] not caching incomplete or failing baseline result")

    test = run_engine(args, test_engine, "test", args.ik_server_bin, IK_PORT, out_root, model_family, prompt_override=ref["prompt"])
    comp = compare_rows(ref, test, thresholds, out_root)
    summarize_comparison(comp, out_root)
    return 0 if comp.get("all_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
