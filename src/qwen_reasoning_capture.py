#!/usr/bin/env python3
"""Capture explicit vLLM Qwen teacher traces for distillation.

This is an opt-in, no-tools, non-streaming capture lane. It records the
prompt, separated reasoning, answer, budget/termination diagnostics, token IDs,
and at most five returned alternatives per generated token. It never requests
or stores full-vocabulary logits.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from qwen_budget_policy import resolve_budget  # noqa: E402

MAX_TOP_LOGPROBS = 5
DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024 * 1024

FORWARDED_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "stop",
    "bad_words",
    "chat_template_kwargs",
    "response_format",
    "reasoning_effort",
)


def parse_args() -> argparse.Namespace:
    default_output = Path.home() / ".local/state/qwen-serving-toolkit/vllm-corpus/captures.jsonl"
    parser = argparse.ArgumentParser(
        description="Capture bounded Qwen reasoning traces and top-5 logprobs from vLLM."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL tasks")
    parser.add_argument("--output", type=Path, default=default_output, help="Persistent output JSONL")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI API base URL")
    parser.add_argument("--model", default="qwen3.6-35b-corpus")
    parser.add_argument("--profile", default="corpus", choices=("action", "qwen38fn", "corpus"))
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--session-id", default=None, help="Stable session ID for this capture run")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error record and continue after a failed request",
    )
    return parser.parse_args()


def effort_for_task(task: dict[str, Any]) -> str | None:
    effort = task.get("reasoning_effort")
    if effort is None:
        effort = (task.get("chat_template_kwargs") or {}).get("reasoning_effort")
    return None if effort is None else str(effort).strip().lower()


def explicit_budget(task: dict[str, Any]) -> int | None:
    value = task.get("thinking_token_budget")
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("thinking_token_budget must be an integer") from exc
    if value < 0:
        raise ValueError("thinking_token_budget must be non-negative")
    return value


def build_request(task: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], int | None]:
    messages = task.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("each input record must contain a non-empty messages list")
    if task.get("tools") or task.get("tool_choice") not in (None, "none"):
        raise ValueError("distillation capture requires a no-tools request")

    profile = task.get("profile", args.profile)
    fixed_raw = os.environ.get("QWEN_THINKING_TOKEN_BUDGET")
    fixed_budget = None if fixed_raw is None else int(fixed_raw)
    budget = resolve_budget(
        profile=profile,
        effort=effort_for_task(task),
        explicit_budget=explicit_budget(task),
        fixed_budget=fixed_budget,
    )

    request: dict[str, Any] = {
        "model": task.get("model", args.model),
        "messages": messages,
        "include_reasoning": True,
        "return_token_ids": True,
        "logprobs": args.top_logprobs > 0,
        "stream": False,
        "max_completion_tokens": task.get("max_completion_tokens", args.max_completion_tokens),
    }
    if args.top_logprobs > 0:
        request["top_logprobs"] = min(args.top_logprobs, MAX_TOP_LOGPROBS)
    for field in FORWARDED_FIELDS:
        if field in task:
            request[field] = task[field]
    if budget is not None:
        request["thinking_token_budget"] = budget
    return request, budget


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    session_id: str,
    request_id: str,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Session-ID": session_id,
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    if not isinstance(value, dict):
        raise ValueError("endpoint returned a non-object JSON response")
    return value


def cap_stored_logprobs(response: dict[str, Any]) -> dict[str, Any]:
    """Keep storage bounded even if a backend ignores the requested cap."""
    stored = copy.deepcopy(response)
    for choice in stored.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            continue
        for token in logprobs.get("content") or []:
            if not isinstance(token, dict):
                continue
            alternatives = token.get("top_logprobs")
            if isinstance(alternatives, list):
                token["top_logprobs"] = alternatives[:MAX_TOP_LOGPROBS]
    return stored


def reasoning_and_answer(response: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    return reasoning, message.get("content"), {
        "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens": details.get("reasoning_tokens", usage.get("reasoning_tokens")),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "token_ids": choice.get("token_ids"),
        "logprobs": choice.get("logprobs"),
        "tool_calls": message.get("tool_calls"),
    }


def diagnostics(
    response: dict[str, Any],
    budget: int | None,
    reasoning: Any,
    selected_logprobs: Any,
) -> dict[str, Any]:
    reasoning_tokens = (response.get("usage") or {}).get("reasoning_tokens")
    details = (response.get("usage") or {}).get("completion_tokens_details") or {}
    if reasoning_tokens is None and isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")

    if budget == 0:
        termination = "disabled"
    elif reasoning_tokens is not None and budget is not None and reasoning_tokens >= budget:
        termination = "budget"
    elif reasoning:
        termination = "natural_or_unknown"
    else:
        termination = "none"

    max_top = 0
    token_count = 0
    if isinstance(selected_logprobs, dict):
        content = selected_logprobs.get("content") or []
        if isinstance(content, list):
            token_count = len(content)
            max_top = max(
                (len(item.get("top_logprobs") or []) for item in content if isinstance(item, dict)),
                default=0,
            )
    return {
        "budget_tokens": budget,
        "reasoning_tokens": reasoning_tokens,
        "termination": termination,
        "end_tag": "</think>" if reasoning is not None else None,
        "end_sequence_emitted": None,
        "logprobs_tokens": token_count,
        "max_returned_top_logprobs": max_top,
    }


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    args = parse_args()
    if not 0 <= args.top_logprobs <= MAX_TOP_LOGPROBS:
        raise SystemExit(f"--top-logprobs must be between 0 and {MAX_TOP_LOGPROBS}")
    if args.max_completion_tokens <= 0:
        raise SystemExit("--max-completion-tokens must be positive")
    if args.max_output_bytes <= 0:
        raise SystemExit("--max-output-bytes must be positive")

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session_id = args.session_id or f"vllm-capture-{uuid.uuid4()}"
    completed = 0
    failed = 0

    with args.input.open(encoding="utf-8") as source, args.output.open("a", encoding="utf-8") as sink:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            task: dict[str, Any] = {}
            request_id = f"{session_id}-request-{line_no}-{uuid.uuid4().hex[:8]}"
            try:
                task = json.loads(line)
                if not isinstance(task, dict):
                    raise ValueError("input record must be a JSON object")
                request, budget = build_request(task, args)
                started = time.perf_counter()
                response = post_json(endpoint, request, args.timeout, session_id, request_id)
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                stored_response = cap_stored_logprobs(response)
                reasoning, content, fields = reasoning_and_answer(stored_response)
                record = {
                    "schema": "qwen-reasoning-capture.v2",
                    "backend": "vllm",
                    "capture_id": str(uuid.uuid4()),
                    "source_id": task.get("id"),
                    "captured_at": now_utc(),
                    "source_line": line_no,
                    "latency_ms": latency_ms,
                    "session_id": session_id,
                    "request_id": request_id,
                    "model": stored_response.get("model", request.get("model")),
                    "messages": request["messages"],
                    "reasoning_effort": request.get("reasoning_effort"),
                    "thinking_token_budget": budget,
                    "reasoning": reasoning,
                    "reasoning_content": reasoning,
                    "content": content,
                    "tool_calls": fields["tool_calls"],
                    "token_ids": fields["token_ids"],
                    "logprobs": fields["logprobs"],
                    "diagnostics": diagnostics(stored_response, budget, reasoning, fields["logprobs"]),
                    "request": request,
                    "response": stored_response,
                }
                completed += 1
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError, urllib.error.URLError) as exc:
                failed += 1
                record = {
                    "schema": "qwen-reasoning-capture.v2",
                    "backend": "vllm",
                    "capture_id": str(uuid.uuid4()),
                    "source_id": task.get("id"),
                    "captured_at": now_utc(),
                    "source_line": line_no,
                    "session_id": session_id,
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if not args.continue_on_error:
                    encoded = json.dumps(record, ensure_ascii=False) + "\n"
                    if sink.tell() + len(encoded.encode("utf-8")) > args.max_output_bytes:
                        raise RuntimeError("output quota reached")
                    sink.write(encoded)
                    sink.flush()
                    raise

            encoded = json.dumps(record, ensure_ascii=False) + "\n"
            encoded_bytes = len(encoded.encode("utf-8"))
            if sink.tell() + encoded_bytes > args.max_output_bytes:
                raise RuntimeError(
                    f"output quota reached ({args.max_output_bytes} bytes); curate or rotate {args.output}"
                )
            sink.write(encoded)
            sink.flush()
            print(f"captured={completed} failed={failed} line={line_no}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
