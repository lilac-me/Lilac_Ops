#!/usr/bin/env python3
"""Compare input token lengths between a bad and a good vLLM-Ascend service.

Used in the C (dataset accuracy drop) branch, right after the AISBench run:
before blaming weights or operators, check whether both sides actually fed the
same input. A *constant* per-sample delta between bad and good input token
counts points at a configuration mismatch (chat template / system prompt / BOS /
add_generation_prompt) rather than at a numerical defect.

Two ways to feed it:

  record mode : --bad FILE --good FILE       (JSONL/JSON, prompt + token length)
  replay mode : --bad-url URL --good-url URL --prompts FILE

Per-pair details go to stdout as JSONL; the verdict goes to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PROMPT_KEYS = ("prompt", "origin_prompt", "question", "input", "text", "query")
LENGTH_KEYS = (
    "input_tokens",
    "input_token_len",
    "prompt_tokens",
    "num_input_tokens",
    "input_len",
    "input_length",
)
VERDICTS = {
    "aligned": "input is aligned, continue with the bad-case analysis",
    "template-offset": "constant offset -> chat template / prompt config is likely "
                       "NOT aligned; check the template before blaming the model",
    "tokenizer-mismatch": "same length but different token ids -> tokenizer / vocab / "
                          "special-token difference",
    "input-not-comparable": "deltas are not constant -> dataset construction or tokenizer "
                            "differs per sample, not a fixed template prefix",
    "insufficient": "not enough matched pairs, collect more samples before concluding",
}


def load_records(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    records = []
    if path.endswith((".jsonl", ".ndjson")):
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    else:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    records = value
                    break
            if not records:
                records = [parsed]
    return [r for r in records if isinstance(r, dict)]


def prompt_of(record):
    for key in PROMPT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for item in messages:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        joined = "\n".join(p for p in parts if p)
        if joined.strip():
            return joined
    return None


def lookup(record, dotted):
    current = record
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def length_of(record, length_field):
    if length_field:
        value = lookup(record, length_field)
        if isinstance(value, int) and value >= 0:
            return value
        raise SystemExit(
            f"field '{length_field}' not found or not an integer in record: {record}")
    for key in LENGTH_KEYS:
        value = record.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    usage = record.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "input_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
    return None


def prompt_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(
                response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, {"error": body[:500]}
    except Exception as exc:  # noqa: BLE001 - report and keep the run going
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def replay_one(base_url, endpoint, model, text, timeout, return_token_ids):
    url = f"{base_url.rstrip('/')}{endpoint}"
    if "chat" in endpoint:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": text}]}
    else:
        payload = {"model": model, "prompt": text}
    payload.update({"max_tokens": 1, "temperature": 0})
    if return_token_ids:
        payload["return_token_ids"] = True
    status, parsed = post_json(url, payload, timeout)
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    tokens = None
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "input_tokens"):
            if isinstance(usage.get(key), int):
                tokens = usage[key]
                break
    prompt_ids = parsed.get("prompt_token_ids") if isinstance(parsed, dict) else None
    if tokens is None and isinstance(prompt_ids, list):
        tokens = len(prompt_ids)
    return {
        "status": status,
        "input_tokens": tokens,
        "prompt_token_ids": prompt_ids if isinstance(prompt_ids, list) else None,
        "error": None if tokens is not None else (
            str(parsed.get("error")) if isinstance(parsed, dict) else "no usage in response"),
    }


def first_diff(left, right):
    if not left or not right:
        return None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def common_span(left, right):
    """Longest common prefix and suffix lengths of two token id lists."""
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(left), len(right)) - prefix
           and left[-1 - suffix] == right[-1 - suffix]):
        suffix += 1
    return prefix, suffix


def maybe_decode(tokenizer, ids):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:  # noqa: BLE001
        return None


def load_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print("[warn] transformers not importable, token segments stay as ids",
              file=sys.stderr)
        return None
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tokenizer load failed ({exc}), segments stay as ids", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare bad/good input token lengths for the same prompts.")
    parser.add_argument("--bad", help="bad-side JSONL/JSON with prompt + length")
    parser.add_argument("--good", help="good-side JSONL/JSON with prompt + length")
    parser.add_argument("--bad-url", help="bad-side base URL, e.g. http://127.0.0.1:8106")
    parser.add_argument("--good-url", help="good-side base URL")
    parser.add_argument("--prompts", help="prompts JSONL/JSON (or an AISBench result file)")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--length-field", help="dotted field holding the input length")
    parser.add_argument("--limit", type=int, default=50,
                        help="max prompts to replay (0 = all). Default: 50")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--dominance", type=float, default=0.9,
                        help="share of pairs that must share one delta. Default: 0.9")
    parser.add_argument("--tokenizer", help="model path, decodes differing token segments")
    parser.add_argument("--json-out", help="write the summary as JSON to this path")
    parser.add_argument("--no-return-token-ids", dest="return_token_ids",
                        action="store_false",
                        help="do not send return_token_ids when replaying")
    parser.set_defaults(return_token_ids=True)
    args = parser.parse_args()

    replay = bool(args.bad_url or args.good_url)
    if replay:
        if not (args.bad_url and args.good_url and args.prompts):
            parser.error("replay mode needs --bad-url, --good-url and --prompts")
    else:
        if not (args.bad and args.good):
            parser.error("record mode needs --bad and --good (or use replay mode)")

    tokenizer = load_tokenizer(args.tokenizer)
    pairs = []
    unmatched = 0

    if replay:
        prompts = []
        for record in load_records(args.prompts):
            text = prompt_of(record)
            if text:
                prompts.append(text)
        if args.limit:
            prompts = prompts[: args.limit]
        print(f"[replay] prompts={len(prompts)} endpoint={args.endpoint}",
              file=sys.stderr)

        def run(text):
            bad = replay_one(args.bad_url, args.endpoint, args.model, text,
                             args.timeout, args.return_token_ids)
            good = replay_one(args.good_url, args.endpoint, args.model, text,
                              args.timeout, args.return_token_ids)
            return text, bad, good

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for text, bad, good in pool.map(run, prompts):
                if bad["input_tokens"] is None or good["input_tokens"] is None:
                    unmatched += 1
                    pairs.append({"key": prompt_key(text), "error":
                                  f"bad={bad['error']} good={good['error']}"})
                    continue
                pairs.append({
                    "key": prompt_key(text),
                    "prompt_head": text[:60],
                    "good_len": good["input_tokens"],
                    "bad_len": bad["input_tokens"],
                    "good_ids": good["prompt_token_ids"],
                    "bad_ids": bad["prompt_token_ids"],
                })
    else:
        def index(path):
            table = {}
            for record in load_records(path):
                text = prompt_of(record)
                if not text:
                    continue
                try:
                    table[prompt_key(text)] = length_of(record, args.length_field)
                except SystemExit as exc:
                    raise SystemExit(f"{path}: {exc}")
            return table

        bad_table, good_table = index(args.bad), index(args.good)
        for key, bad_len in bad_table.items():
            good_len = good_table.get(key)
            if good_len is None:
                unmatched += 1
                continue
            pairs.append({"key": key, "good_len": good_len, "bad_len": bad_len})
        print(f"[record] bad={len(bad_table)} good={len(good_table)} "
              f"matched={len(pairs)} unmatched={unmatched}", file=sys.stderr)

    usable = [p for p in pairs if "error" not in p]
    for pair in pairs:
        if "error" in pair:
            print(json.dumps({"key": pair["key"], "error": pair["error"]},
                             ensure_ascii=False))
            continue
        delta = pair["bad_len"] - pair["good_len"]
        diff = first_diff(pair.get("good_ids"), pair.get("bad_ids"))
        pair["delta"] = delta
        pair["first_diff"] = diff
        print(json.dumps({
            "key": pair["key"], "good_len": pair["good_len"],
            "bad_len": pair["bad_len"], "delta": delta,
            "first_diff": diff, "prompt_head": pair.get("prompt_head"),
        }, ensure_ascii=False))

    deltas = [p["delta"] for p in usable]
    counts = Counter(deltas)
    comparable = [p for p in usable if p.get("good_ids") and p.get("bad_ids")]
    differing = [p for p in comparable if p.get("first_diff") is not None]

    if len(deltas) < args.min_pairs:
        verdict = "insufficient"
    elif not deltas:
        verdict = "insufficient"
    else:
        dominant, dominant_count = counts.most_common(1)[0]
        share = dominant_count / len(deltas)
        if dominant == 0:
            if share < args.dominance:
                verdict = "input-not-comparable"
            elif differing:
                verdict = "tokenizer-mismatch"
            else:
                verdict = "aligned"
        else:
            verdict = "template-offset" if share >= args.dominance else "input-not-comparable"

    summary = {
        "mode": "replay" if replay else "record",
        "pairs": len(pairs),
        "usable": len(usable),
        "unmatched": unmatched,
        "distinct_deltas": dict(counts),
        "verdict": verdict,
        "verdict_hint": VERDICTS[verdict],
    }

    if deltas:
        summary["good_len_range"] = [min(p["good_len"] for p in usable),
                                     max(p["good_len"] for p in usable)]
        summary["bad_len_range"] = [min(p["bad_len"] for p in usable),
                                    max(p["bad_len"] for p in usable)]
    if comparable:
        summary["token_ids_comparable"] = len(comparable)
        summary["token_ids_differing"] = len(differing)
    if differing:
        sample = differing[0]
        left, right = sample["good_ids"], sample["bad_ids"]
        prefix, suffix = common_span(left, right)
        summary["first_diff_sample"] = {
            "key": sample["key"],
            "good_len": sample["good_len"],
            "bad_len": sample["bad_len"],
            "first_diff": sample["first_diff"],
            "common_prefix": prefix,
            "common_suffix": suffix,
            "good_segment": left[prefix:len(left) - suffix if suffix else len(left)],
            "bad_segment": right[prefix:len(right) - suffix if suffix else len(right)],
        }
        decoded_good = maybe_decode(tokenizer, summary["first_diff_sample"]["good_segment"])
        decoded_bad = maybe_decode(tokenizer, summary["first_diff_sample"]["bad_segment"])
        if decoded_good is not None:
            summary["first_diff_sample"]["good_text"] = decoded_good
        if decoded_bad is not None:
            summary["first_diff_sample"]["bad_text"] = decoded_bad

    print("\n=== input token check ===", file=sys.stderr)
    print(f"mode={summary['mode']} pairs={summary['pairs']} usable={summary['usable']} "
          f"unmatched={unmatched}", file=sys.stderr)
    if "good_len_range" in summary:
        print(f"length: good {summary['good_len_range'][0]}..{summary['good_len_range'][1]}  "
              f"bad {summary['bad_len_range'][0]}..{summary['bad_len_range'][1]}",
              file=sys.stderr)
    if deltas:
        dominant, dominant_count = counts.most_common(1)[0]
        print(f"delta: most common {dominant:+d} in {dominant_count}/{len(deltas)} "
              f"({dominant_count / len(deltas):.1%})  distinct={len(counts)}",
              file=sys.stderr)
    if comparable:
        print(f"prompt_token_ids: comparable={len(comparable)} differing={len(differing)}",
              file=sys.stderr)
    if "first_diff_sample" in summary:
        sample = summary["first_diff_sample"]
        print(f"first diff: token#{sample['first_diff']} "
              f"(common prefix={sample['common_prefix']}, suffix={sample['common_suffix']})",
              file=sys.stderr)
        print(f"  good segment ids={sample['good_segment'][:16]}"
              f"{' len=' + str(len(sample['good_segment'])) if len(sample['good_segment']) > 16 else ''}",
              file=sys.stderr)
        print(f"  bad  segment ids={sample['bad_segment'][:16]}"
              f"{' len=' + str(len(sample['bad_segment'])) if len(sample['bad_segment']) > 16 else ''}",
              file=sys.stderr)
        for side in ("good", "bad"):
            if f"{side}_text" in sample:
                print(f"  {side} text: {sample[f'{side}_text']!r}", file=sys.stderr)
    print(f"verdict={verdict}", file=sys.stderr)
    print(f"hint: {VERDICTS[verdict]}", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
