#!/usr/bin/env python3
"""MinHash contamination filtering for training corpus.

Builds MinHash signatures from evaluation samples and filters training documents
that are near-duplicates above a similarity threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasketch import MinHash, MinHashLSH

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class FilterStats:
    train_total: int = 0
    train_kept: int = 0
    train_removed: int = 0
    eval_total: int = 0
    eval_used: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter contaminated train samples with MinHash")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/phase1_v2_de_eval/train_corpus.jsonl"),
        help="Input train jsonl file",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/processed/phase1_v2_de_eval/eval_mcq_global_mmlu_de.jsonl"),
        help="Input eval jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/phase1_v2_de_eval_minhash"),
        help="Directory for filtered train file and contamination report",
    )
    parser.add_argument("--threshold", type=float, default=0.8, help="MinHash Jaccard threshold for removal")
    parser.add_argument("--ngram-size", type=int, default=3, help="Token n-gram size for shingles")
    parser.add_argument("--num-perm", type=int, default=128, help="Number of MinHash permutations")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means all eval samples")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means all train samples")
    parser.add_argument("--sample-matches", type=int, default=25, help="Store up to N removed examples in report")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress every N train rows (0 disables progress logs)",
    )
    parser.add_argument(
        "--eval-progress-every",
        type=int,
        default=200,
        help="Print progress every N eval rows while building MinHash index (0 disables progress logs)",
    )
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows and idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def minhash_from_text(text: str, ngram_size: int, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for gram in ngrams(tokenize(text), ngram_size):
        mh.update(gram.encode("utf-8"))
    return mh


def eval_text_from_row(row: Dict) -> str:
    # Supports Global-MMLU style rows and falls back to common text-like fields.
    if "question" in row and "option_a" in row and "option_b" in row:
        options = [row.get("option_a", ""), row.get("option_b", ""), row.get("option_c", ""), row.get("option_d", "")]
        return " ".join([row.get("question", "")] + [o for o in options if o])

    for key in ("text", "question", "prompt", "article"):
        if key in row and row[key]:
            return str(row[key])

    return " ".join(str(v) for v in row.values() if isinstance(v, (str, int, float)))


def train_text_from_row(row: Dict) -> str:
    return str(row.get("text", ""))


def build_eval_index(
    eval_rows: List[Dict],
    ngram_size: int,
    num_perm: int,
    threshold: float,
    progress_every: int,
) -> Tuple[MinHashLSH, Dict[str, MinHash]]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: Dict[str, MinHash] = {}
    started = time.time()
    total = len(eval_rows)

    for i, row in enumerate(eval_rows):
        text = eval_text_from_row(row)
        if not text.strip():
            continue
        key = f"eval_{i}"
        mh = minhash_from_text(text, ngram_size=ngram_size, num_perm=num_perm)
        signatures[key] = mh
        lsh.insert(key, mh)

        if progress_every > 0 and ((i + 1) % progress_every == 0 or (i + 1) == total):
            elapsed = time.time() - started
            speed = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[eval-index] {i + 1}/{total} rows, {speed:.1f} rows/s")

    return lsh, signatures


def main() -> None:
    args = parse_args()

    stats = FilterStats()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_jsonl(args.eval_file, max_rows=args.max_eval_samples)
    train_rows = read_jsonl(args.train_file, max_rows=args.max_train_samples)

    stats.eval_total = len(eval_rows)
    stats.train_total = len(train_rows)

    print("Building eval MinHash index...")
    lsh, eval_signatures = build_eval_index(
        eval_rows=eval_rows,
        ngram_size=args.ngram_size,
        num_perm=args.num_perm,
        threshold=args.threshold,
        progress_every=args.eval_progress_every,
    )
    stats.eval_used = len(eval_signatures)

    print("Filtering train corpus...")
    kept_rows: List[Dict] = []
    removed_examples: List[Dict] = []
    started = time.time()
    total = len(train_rows)

    for i, row in enumerate(train_rows):
        text = train_text_from_row(row)
        if not text.strip():
            kept_rows.append(row)
            continue

        mh = minhash_from_text(text, ngram_size=args.ngram_size, num_perm=args.num_perm)
        candidates = lsh.query(mh)

        is_contaminated = False
        best_score = 0.0
        best_key = None

        for key in candidates:
            score = mh.jaccard(eval_signatures[key])
            if score > best_score:
                best_score = score
                best_key = key
            if score >= args.threshold:
                is_contaminated = True
                break

        if is_contaminated:
            stats.train_removed += 1
            if len(removed_examples) < args.sample_matches:
                removed_examples.append(
                    {
                        "train_index": i,
                        "eval_key": best_key,
                        "similarity": round(best_score, 4),
                        "train_text_preview": text[:300],
                    }
                )
        else:
            kept_rows.append(row)

        processed = i + 1
        if args.progress_every > 0 and (processed % args.progress_every == 0 or processed == total):
            elapsed = time.time() - started
            speed = processed / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total - processed)
            eta_sec = int(remaining / speed) if speed > 0 else -1
            eta = "unknown" if eta_sec < 0 else f"{eta_sec}s"
            print(
                f"[filter] {processed}/{total} rows, kept={len(kept_rows)}, "
                f"removed={stats.train_removed}, speed={speed:.1f} rows/s, eta={eta}"
            )

    stats.train_kept = len(kept_rows)

    filtered_train_path = output_dir / "train_corpus.filtered.jsonl"
    write_jsonl(filtered_train_path, kept_rows)

    report = {
        "stats": asdict(stats),
        "config": {
            "train_file": str(args.train_file),
            "eval_file": str(args.eval_file),
            "output_dir": str(args.output_dir),
            "threshold": args.threshold,
nano-alpha-llm-pretrain/notebooks            "ngram_size": args.ngram_size,
            "num_perm": args.num_perm,
            "max_eval_samples": args.max_eval_samples,
            "max_train_samples": args.max_train_samples,
            "sample_matches": args.sample_matches,
            "progress_every": args.progress_every,
            "eval_progress_every": args.eval_progress_every,
        },
        "artifacts": {
            "filtered_train_file": str(filtered_train_path),
        },
        "removed_examples": removed_examples,
    }

    report_path = output_dir / "minhash_contamination_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Train total:   {stats.train_total}")
    print(f"Train kept:    {stats.train_kept}")
    print(f"Train removed: {stats.train_removed}")
    print(f"Eval used:     {stats.eval_used}")
    print(f"Filtered file: {filtered_train_path}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
