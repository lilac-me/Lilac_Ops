#!/usr/bin/env python3
"""Create one fixed Transformers-tokenized input payload for precision alignment."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from .alignment_utils import FORMAT_VERSION, input_sha256, normalize_fixed_input
except ImportError:  # Direct script execution.
    from alignment_utils import FORMAT_VERSION, input_sha256, normalize_fixed_input  # type: ignore


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args(
    *,
    default_output: str = "alignment_input.pt",
    model_label: str = "Transformers",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        "--model-dir",
        dest="tokenizer_dir",
        required=True,
        help=f"Local {model_label} tokenizer/model directory.",
    )
    parser.add_argument("--output-file", default=default_output)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt",
        action="append",
        help="Prompt text. Repeat the option to create a batch.",
    )
    prompts.add_argument(
        "--prompts-file",
        help="UTF-8 text file containing one non-empty prompt per line.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use tokenizer.encode instead of the tokenizer chat template.",
    )
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_true",
        help="Do not append the assistant-generation prefix in chat-template mode.",
    )
    parser.add_argument(
        "--no-add-special-tokens",
        action="store_true",
        help="Do not add tokenizer special tokens in raw-text mode.",
    )
    parser.add_argument(
        "--fail-on-truncation",
        action="store_true",
        help="Reject prompts longer than --max-length instead of recording truncation.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-remote-files",
        action="store_true",
        help="Permit model/tokenizer files not already available locally.",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace, default_prompt: str | None) -> list[str]:
    if args.prompt:
        return args.prompt
    if args.prompts_file:
        prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError(f"No non-empty prompts found in {args.prompts_file}")
        return prompts
    if default_prompt is None:
        raise ValueError("Provide --prompt or --prompts-file; no model default is configured.")
    return [default_prompt]


def normalize_token_ids(value: Any) -> list[int]:
    """Normalize one tokenizer result across Transformers versions."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise KeyError(
                "Tokenizer output is a mapping without input_ids; "
                f"available keys: {list(value.keys())}"
            )
        value = value["input_ids"]

    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.ndim != 1:
            raise TypeError(
                "Expected one tokenized conversation with shape [S] or [1,S], "
                f"got Tensor{tuple(value.shape)}"
            )
        value = value.to(torch.long).tolist()

    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]

    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, Integral) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"Unexpected tokenizer result: {type(value)!r}")
    if not value:
        raise ValueError("A prompt produced no input IDs.")
    return [int(item) for item in value]


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
    add_generation_prompt: bool,
    add_special_tokens: bool,
) -> list[int]:
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("The tokenizer has no chat_template; pass --no-chat-template.")
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        result = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    return normalize_token_ids(result)


def ensure_pad_token(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token_id = int(eos_token_id)
    return int(tokenizer.pad_token_id)


def build_fixed_input_payload(
    tokenizer: Any,
    prompts: list[str],
    *,
    tokenizer_dir: str,
    max_length: int,
    use_chat_template: bool,
    add_generation_prompt: bool,
    add_special_tokens: bool,
    fail_on_truncation: bool,
) -> tuple[dict[str, Any], list[int], list[bool]]:
    if max_length <= 0:
        raise ValueError("--max-length must be positive")
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("At least one non-empty prompt is required.")

    tokenizer.padding_side = "right"
    pad_token_id = ensure_pad_token(tokenizer)
    encoded = [
        encode_prompt(
            tokenizer,
            prompt,
            use_chat_template=use_chat_template,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens,
        )
        for prompt in prompts
    ]
    original_lengths = [len(token_ids) for token_ids in encoded]
    truncated = [length > max_length for length in original_lengths]
    if fail_on_truncation and any(truncated):
        raise ValueError(
            "At least one prompt exceeds --max-length: "
            f"lengths={original_lengths}, max_length={max_length}."
        )
    encoded = [token_ids[:max_length] for token_ids in encoded]
    valid_lengths = [len(token_ids) for token_ids in encoded]
    padded_length = max(valid_lengths)

    input_ids = torch.full(
        (len(encoded), padded_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, token_ids in enumerate(encoded):
        length = len(token_ids)
        input_ids[row, :length] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[row, :length] = 1

    digest = input_sha256(input_ids, attention_mask)
    tokenizer_path = Path(tokenizer_dir)
    tokenizer_identity = (
        str(tokenizer_path.resolve()) if tokenizer_path.exists() else tokenizer_dir
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_sha256": digest,
        "metadata": {
            "source": "transformers-tokenizer",
            "tokenizer_dir": tokenizer_identity,
            "tokenizer_dir_or_id": tokenizer_dir,
            "tokenizer_class": tokenizer.__class__.__name__,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_chat_template": use_chat_template,
            "add_generation_prompt": add_generation_prompt,
            "add_special_tokens": add_special_tokens,
            "prompts": prompts,
            "original_lengths": original_lengths,
            "valid_lengths": valid_lengths,
            "max_length": max_length,
            "truncated": truncated,
        },
    }
    normalized_ids, normalized_mask, normalized_digest = normalize_fixed_input(payload)
    if not torch.equal(normalized_ids, input_ids) or not torch.equal(
        normalized_mask, attention_mask
    ):
        raise RuntimeError("Generated input payload changed during contract validation.")
    if normalized_digest != digest:
        raise RuntimeError("Generated input digest changed during contract validation.")
    return payload, valid_lengths, truncated


def main(
    *,
    default_prompt: str | None = None,
    default_output: str = "alignment_input.pt",
    model_label: str = "Transformers",
) -> None:
    args = parse_args(default_output=default_output, model_label=model_label)
    prompts = load_prompts(args, default_prompt)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.allow_remote_files,
    )
    payload, valid_lengths, truncated = build_fixed_input_payload(
        tokenizer,
        prompts,
        tokenizer_dir=args.tokenizer_dir,
        max_length=args.max_length,
        use_chat_template=not args.no_chat_template,
        add_generation_prompt=not args.no_add_generation_prompt,
        add_special_tokens=not args.no_add_special_tokens,
        fail_on_truncation=args.fail_on_truncation,
    )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "shape": list(payload["input_ids"].shape),
                "valid_lengths": valid_lengths,
                "truncated": truncated,
                "input_sha256": payload["input_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
