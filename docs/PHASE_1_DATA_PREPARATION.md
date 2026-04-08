# Phase 1: Data Preparation and Contamination Control

## Definition
Prepare, clean, validate, and freeze train/validation corpora with contamination controls.

## What We Implement
1. Data ingestion and canonical JSONL outputs
2. Split generation and metadata tracking
3. MinHash-based contamination filtering against evaluation-oriented samples
4. Reports, manifests, and checksums for reproducibility

## Why It Matters
1. Dataset quality strongly affects model quality.
2. Data leakage can invalidate later benchmark conclusions.
3. Frozen artifacts are required for fair checkpoint/run comparisons.

## What It Evaluates
1. Corpus integrity and schema consistency
2. Contamination risk reduction effectiveness
3. Reproducibility of dataset construction

## Typical Outputs
1. train_corpus.jsonl
2. val_corpus.jsonl
3. contamination report
4. checksums and preparation report
