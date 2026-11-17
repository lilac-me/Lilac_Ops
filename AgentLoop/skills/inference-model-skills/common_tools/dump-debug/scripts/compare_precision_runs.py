#!/usr/bin/env python3
"""Run repeated vLLM completion requests and report output consistency.

The determinism verdict is based on content (text + token ids); logprob drift is
reported separately so floating-point jitter cannot mask a content mismatch.
token ids require the server to accept ``return_token_ids: true`` (vLLM >= 0.10).
Without them the comparison degrades to text only and the first differing token
position cannot be reported.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request


def digest(payload):
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def request_completion(args, temperature, run_index):
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": temperature,
    }
    if args.logprobs is not None:
        payload["logprobs"] = args.logprobs
    if args.return_token_ids:
        payload["return_token_ids"] = True
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}{args.endpoint}",
        data=data,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except Exception as exc:
        return {
            "label": args.label,
            "temperature": temperature,
            "run": run_index,
            "status": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None

    choice = None
    if isinstance(parsed, dict) and parsed.get("choices"):
        choice = parsed["choices"][0]
    text = choice.get("text", "") if isinstance(choice, dict) else ""
    token_ids = choice.get("token_ids") if isinstance(choice, dict) else None
    logprobs = None
    if isinstance(choice, dict) and isinstance(choice.get("logprobs"), dict):
        logprobs = choice["logprobs"].get("token_logprobs")

    return {
        "label": args.label,
        "temperature": temperature,
        "run": run_index,
        "status": status,
        "ok": bool(choice),
        "content_hash": digest({"text": text, "token_ids": token_ids}),
        "logprob_hash": digest({"logprobs": logprobs}),
        "has_token_ids": token_ids is not None,
        "text": text,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "error": None if choice else body[:1000],
    }


def first_difference(reference, candidate):
    """Describe the first content difference, or return None when identical."""
    ref_ids = reference.get("token_ids")
    cand_ids = candidate.get("token_ids")
    if ref_ids and cand_ids:
        for index, (left, right) in enumerate(zip(ref_ids, cand_ids)):
            if left != right:
                return f"token#{index}"
        if len(ref_ids) != len(cand_ids):
            return f"token#{min(len(ref_ids), len(cand_ids))}(length)"
        return None
    ref_text = reference.get("text") or ""
    cand_text = candidate.get("text") or ""
    for index, (left, right) in enumerate(zip(ref_text, cand_text)):
        if left != right:
            return f"char#{index}"
    if len(ref_text) != len(cand_text):
        return f"char#{min(len(ref_text), len(cand_text))}(length)"
    return None


def logprob_drift(reference, candidate):
    """Return max |delta| between token logprobs, or None when not comparable."""
    left = reference.get("logprobs") or []
    right = candidate.get("logprobs") or []
    if not left or not right or len(left) != len(right):
        return None
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8106")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--api-key")
    parser.add_argument("--no-return-token-ids", dest="return_token_ids",
                        action="store_false",
                        help="do not send return_token_ids (server without support); "
                             "the content comparison degrades to text only")
    parser.set_defaults(return_token_ids=True)
    args = parser.parse_args()

    results = []
    for temperature in args.temperatures:
        for run_index in range(1, args.repeats + 1):
            result = request_completion(args, temperature, run_index)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)

    missing_token_ids = any(
        r.get("ok") and r.get("token_ids") is None for r in results)
    if args.return_token_ids and missing_token_ids:
        print("[warn] responses carry no token_ids: the server ignored or rejected "
              "return_token_ids (vLLM < 0.10, or disabled). The content comparison "
              "degrades to text only; the first differing token position is "
              "unavailable.", file=sys.stderr)

    print("\n=== summary ===", file=sys.stderr)
    for temperature in args.temperatures:
        group = [r for r in results if r["temperature"] == temperature]
        succeeded = [r for r in group if r.get("ok")]
        failures = len(group) - len(succeeded)
        reference = succeeded[0] if succeeded else None
        diff_at = None
        drift = 0.0
        for candidate in succeeded[1:]:
            if diff_at is None:
                diff_at = first_difference(reference, candidate)
            pair_drift = logprob_drift(reference, candidate)
            if pair_drift is None:
                drift = None
            elif drift is not None:
                drift = max(drift, pair_drift)

        if failures:
            content = "incomplete"
        elif diff_at:
            content = f"diff@{diff_at}"
        else:
            content = "stable"
        if temperature == 0.0:
            verdict = "deterministic" if content == "stable" else "NON-DETERMINISTIC"
        else:
            verdict = "stable" if content == "stable" else "varies"
        drift_text = "n/a" if drift is None else f"{drift:.3g}"
        print(
            f"temperature={temperature}: runs={len(group)} ok={len(succeeded)} "
            f"content={content} logprob_drift={drift_text} verdict={verdict}",
            file=sys.stderr,
        )
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
