#!/usr/bin/env python3
"""Train a BPE tokenizer for Phase 2 pretraining.

Reads JSONL rows from the prepared training corpus and trains a 32k-token
byte-level BPE tokenizer with common special tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on phase1 corpus")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/phase1_v2_de_eval/train_corpus.jsonl"),
        help="Input training corpus JSONL path",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="JSON field containing text",
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="nano_alpha_bpe",
        help="Tokenizer file prefix name",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print tokenizer iterator progress every N rows (1 prints every row)",
    )
    return parser.parse_args()


def iter_jsonl_text(path: Path, text_field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(text_field, "")).strip()
            if text:
                yield text


def iter_jsonl_text_with_progress(path: Path, text_field: str, progress_every: int) -> Iterator[str]:
    started = time.time()
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(text_field, "")).strip()
            if not text:
                continue
            count += 1
            if progress_every > 0 and (count % progress_every == 0):
                elapsed = time.time() - started
                speed = count / elapsed if elapsed > 0 else 0.0
                print(f"[tokenizer] row={count} speed={speed:.1f} rows/s")
            yield text

    elapsed = time.time() - started
    speed = count / elapsed if elapsed > 0 else 0.0
    print(f"[tokenizer] completed rows={count} elapsed={elapsed:.1f}s avg_speed={speed:.1f} rows/s")


def main() -> None:
    args = parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        iter_jsonl_text_with_progress(args.train_file, args.text_field, args.progress_every),
        trainer=trainer,
    )

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Special token ids missing after training")

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    tokenizer_json_path = args.output_dir / f"{args.name}.tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    vocab_path = None
    merges_path = None
    model = tokenizer.model
    if hasattr(model, "save"):
        saved = model.save(str(args.output_dir), args.name)
        if len(saved) >= 2:
            vocab_path = Path(saved[0])
            merges_path = Path(saved[1])

    summary = {
        "train_file": str(args.train_file),
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "text_field": args.text_field,
        "progress_every": args.progress_every,
        "tokenizer_json": str(tokenizer_json_path),
        "vocab_file": str(vocab_path) if vocab_path else None,
        "merges_file": str(merges_path) if merges_path else None,
        "special_tokens": special_tokens,
    }

    summary_path = args.output_dir / "tokenizer_training_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Tokenizer training complete.")
    print(f"Tokenizer JSON: {tokenizer_json_path}")
    print(f"Vocab file:     {vocab_path}")
    print(f"Merges file:    {merges_path}")
    print(f"Report:         {summary_path}")


if __name__ == "__main__":
    main()
