#!/usr/bin/env python3
"""Phase 1 data pipeline bootstrap.

Downloads small EN/DE Wikipedia subsets, builds a cleaned corpus,
creates train/validation splits, and exports a small eval set.
"""

from __future__ import annotations

import argparse
from itertools import islice
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset


WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PipelineStats:
    de_loaded: int = 0
    en_loaded: int = 0
    de_kept: int = 0
    en_kept: int = 0
    combined_kept: int = 0
    train_count: int = 0
    val_count: int = 0
    eval_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EN/DE training corpus and eval data")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file containing HF token values",
    )
    parser.add_argument(
        "--train-source",
        type=str,
        choices=["wiki", "opus100"],
        default="wiki",
        help="Training corpus source: wiki (closer to project plan) or opus100 (lighter smoke mode)",
    )
    parser.add_argument(
        "--wiki-dataset",
        type=str,
        default="wikimedia/wikipedia",
        help="HF dataset id for Wikipedia content",
    )
    parser.add_argument(
        "--wiki-date",
        type=str,
        default="20231101",
        help="Wikipedia snapshot date used in config names, e.g. 20231101",
    )
    parser.add_argument("--de-max-docs", type=int, default=20000, help="German Wikipedia docs to load")
    parser.add_argument("--en-max-docs", type=int, default=20000, help="English Wikipedia docs to load")
    parser.add_argument("--eval-max-samples", type=int, default=500, help="Eval examples to export")
    parser.add_argument(
        "--eval-output-name",
        type=str,
        default="eval_mcq.jsonl",
        help="File name for exported eval samples",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="x_csqa",
        help="HF dataset id for eval; if unavailable, eval export will be empty",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="de",
        help="Config/subset for eval dataset",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        help="Split name for eval dataset",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio")
    parser.add_argument("--min-chars", type=int, default=200, help="Minimum characters per kept document")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for jsonl files and report",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def iter_clean_wiki_records(raw_records: Iterable[Dict], lang: str, min_chars: int) -> List[Dict[str, str]]:
    seen = set()
    kept: List[Dict[str, str]] = []
    for row in raw_records:
        text = normalize_text(row.get("text", ""))
        if len(text) < min_chars:
            continue
        text_hash = hash(text)
        if text_hash in seen:
            continue
        seen.add(text_hash)
        kept.append({"lang": lang, "text": text})
    return kept


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_stream_slice(dataset_name: str, config_name: str, split: str, max_rows: int):
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    return list(islice(ds, max_rows))


def load_env_file(env_file: Path) -> bool:
    if not env_file.exists():
        return False

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = hf_token
    if hf_token and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    return True


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    env_file = args.env_file
    if not env_file.is_absolute():
        env_file = Path.cwd() / env_file
    loaded_env = load_env_file(env_file)
    if loaded_env:
        print(f"Loaded env vars from {env_file}")
    else:
        print(f"No env file found at {env_file}; continuing without local token file")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = PipelineStats()

    if args.train_source == "wiki":
        print("Loading Wikipedia subsets (streaming)...")
        de_config = f"{args.wiki_date}.de"
        en_config = f"{args.wiki_date}.en"
        de_raw = load_stream_slice(args.wiki_dataset, de_config, "train", args.de_max_docs)
        en_raw = load_stream_slice(args.wiki_dataset, en_config, "train", args.en_max_docs)
    else:
        print("Loading OPUS-100 de-en subset (lightweight smoke source)...")
        opus_n = max(args.de_max_docs, args.en_max_docs)
        opus_ds = load_dataset("Helsinki-NLP/opus-100", "de-en", split=f"train[:{opus_n}]")
        de_raw = [{"text": row["translation"]["de"]} for row in opus_ds.select(range(min(args.de_max_docs, len(opus_ds))))]
        en_raw = [{"text": row["translation"]["en"]} for row in opus_ds.select(range(min(args.en_max_docs, len(opus_ds))))]

    stats.de_loaded = len(de_raw)
    stats.en_loaded = len(en_raw)

    print("Cleaning and filtering texts...")
    de_rows = iter_clean_wiki_records(de_raw, lang="de", min_chars=args.min_chars)
    en_rows = iter_clean_wiki_records(en_raw, lang="en", min_chars=args.min_chars)

    stats.de_kept = len(de_rows)
    stats.en_kept = len(en_rows)

    combined = de_rows + en_rows
    random.shuffle(combined)
    stats.combined_kept = len(combined)

    val_size = max(1, int(len(combined) * args.val_ratio))
    val_rows = combined[:val_size]
    train_rows = combined[val_size:]

    stats.train_count = len(train_rows)
    stats.val_count = len(val_rows)

    eval_rows: List[Dict] = []
    print(f"Loading eval set ({args.eval_dataset}/{args.eval_config}, streaming)...")
    try:
        eval_ds = load_dataset(args.eval_dataset, args.eval_config, split=args.eval_split, streaming=True)
        eval_rows = [dict(row) for row in islice(eval_ds, args.eval_max_samples)]
    except Exception as exc:
        print(f"Warning: eval dataset load failed: {exc}")
        print("Continuing with empty eval set. You can override --eval-dataset later.")
    stats.eval_count = len(eval_rows)

    print("Writing outputs...")
    write_jsonl(output_dir / "train_corpus.jsonl", train_rows)
    write_jsonl(output_dir / "val_corpus.jsonl", val_rows)
    write_jsonl(output_dir / args.eval_output_name, eval_rows)

    report = {
        "stats": asdict(stats),
        "config": {
            "env_file": str(env_file),
            "train_source": args.train_source,
            "wiki_dataset": args.wiki_dataset,
            "wiki_date": args.wiki_date,
            "de_max_docs": args.de_max_docs,
            "en_max_docs": args.en_max_docs,
            "eval_max_samples": args.eval_max_samples,
            "eval_output_name": args.eval_output_name,
            "eval_dataset": args.eval_dataset,
            "eval_config": args.eval_config,
            "eval_split": args.eval_split,
            "val_ratio": args.val_ratio,
            "min_chars": args.min_chars,
            "seed": args.seed,
        },
        "notes": [
            "MinHash contamination filtering is not applied yet.",
            "If --eval-dataset is unavailable, eval export is intentionally empty for pipeline continuity.",
            "Next step: add n-gram based MinHash overlap checks against eval prompts.",
        ],
    }

    report_path = output_dir / "prepare_data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Train rows: {stats.train_count}")
    print(f"Val rows:   {stats.val_count}")
    print(f"Eval rows:  {stats.eval_count}")
    print(f"Report:     {report_path}")


if __name__ == "__main__":
    main()
