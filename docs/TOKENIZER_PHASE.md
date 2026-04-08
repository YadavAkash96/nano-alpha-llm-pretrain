## Phase 1: Tokenizer Pre-Training (Byte-Level BPE)

This document outlines the execution steps and theoretical methodology behind the custom Byte-Level Byte-Pair Encoding (BBPE) tokenizer trained for the Nano-Alpha pipeline. 

By utilizing BBPE, this pipeline mirrors the tokenization strategies used by modern foundation models like Llama 2, GPT-4, and Aleph Alpha's proprietary models, ensuring efficient cross-lingual processing and zero out-of-vocabulary (`<unk>`) errors.

---

## 1. Execution Steps

To train the tokenizer on your pre-processed JSONL corpus, execute the following script.

**Command:**
```bash
python -u scripts/train_tokenizer.py \
  --train-file data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  --vocab-size 32000 \
  --min-frequency 2 \
  --output-dir artifacts/tokenizer \
  --name nano_alpha_bpe
```

**Expected Outputs:**
Upon completion, the `artifacts/tokenizer/` directory will contain:
1. `nano_alpha_bpe.tokenizer.json`: The full tokenizer configuration.
2. `vocab.json`: The mapping of the 32,000 strings/bytes to integer IDs.
3. `merges.txt`: The ordered list of learned byte-pair combinations.
4. `tokenizer_training_report.json`: A summary of the execution parameters and speeds.

---

## 2. Methodology: Byte-Level BPE Explained

### Why Byte-Level?
Standard character-level tokenizers map Unicode characters to IDs. Because there are over 140,000 Unicode characters, rare symbols (or complex German characters like `ö`) often fall out of the vocabulary, forcing the model to output an `<unk>` (unknown) token. This degrades model performance and introduces blind spots.

Our tokenizer uses `tokenizer.pre_tokenizer = ByteLevel(...)`. This converts all text into raw UTF-8 computer bytes *before* doing anything else. 
* A standard English letter (e.g., `A`) is 1 byte.
* A special character (e.g., `ö`) is converted into 2 bytes: `[195, 182]`.

**Advantage:** The base vocabulary strictly consists of the numbers 0–255. The model will **never** output an `<unk>` because any rare or out-of-vocabulary word can simply decompose back into raw, individual bytes.

### Why BPE?
BPE is a data compression algorithm adapted for NLP. It starts with the base vocabulary (256 bytes) and iteratively merges the most frequently occurring adjacent pairs of tokens in the training corpus. It does this until it reaches the target `vocab_size` (32,000).

---

## 3. Core Rules & Mechanics

The tokenizer relies on strict mathematical and programmatic rules to resolve conflicts and determine when to stop.

* **Target Vocab Size (`32000`)**: The algorithm stops merging when the vocabulary contains exactly 32,000 tokens (256 base bytes + 4 special tokens + 31,740 learned merges).
* **Min Frequency (`2`)**: A pair of tokens must appear *at least* twice in the corpus to be merged. This prevents typos, random hashes, or unique numbers from permanently occupying valuable vocabulary slots.

### Advanced Mechanics: Priority and Tie-Breaking
When observing BPE, you might ask: *How does it decide what to merge first if `[space] + B` and `B + B` appear the exact same number of times in a sequence?*

1. **The Global Context Rule:** The tokenizer does not look at a single sentence; it looks at the entire 2 GB corpus simultaneously. While two pairs might be tied locally in one sentence, global frequencies usually declare a clear winner.
2. **Deterministic Tie-Breaking:** If two pairs have the exact same global frequency (e.g., both appear exactly 42,105 times), the Hugging Face `tokenizers` backend uses a **deterministic lexicographical (alphabetical) sort** based on the byte values. It sorts the tied pairs numerically and picks the lowest byte value first. This guarantees reproducible vocabularies across identical training runs.

---

## 4. End-to-End Algorithmic Example

To understand how BPE actually processes data, we must discard the idea that it reads text "word by word". **BPE is a global greedy algorithm.** It continuously scans the entire corpus, finds the single most frequent pair of tokens globally, merges them everywhere, and repeats.

Let's look at how the algorithm processes three multi-word sequences from the English training data:
1. `" BBC Radio 4"`
2. `" BBC Radio comedy"`
3. `" BBC Radio programmes"`

### Step A: Initial Byte Splitting
First, the `ByteLevel` pre-tokenizer breaks the strings into raw bytes. **Crucially, the space character is treated as just another byte.** *(Represented here as `[space]` for readability).*

Sequence 1 becomes 12 individual byte tokens:
`[space]`, `B`, `B`, `C`, `[space]`, `R`, `a`, `d`, `i`, `o`, `[space]`, `4`

### Step B: The Global Greedy Loop
The algorithm counts every pair of adjacent bytes across the *entire* corpus (including German data, English data, code, etc.). Let's track how `" BBC Radio"` is constructed over the course of thousands of global iterations.

| Global Loop Iteration | The #1 Most Frequent Pair Globally | Impact on our specific string: `" BBC Radio 4"` |
| :--- | :--- | :--- |
| **Start** | (Base 256 bytes) | `[space] B B C [space] R a d i o [space] 4` |
| **Merge 1** | `e` + `r` *(Highly common in DE/EN)* | *(No change to our string)* |
| **Merge 2** | `e` + `n` *(Highly common in DE)* | *(No change to our string)* |
| **... Fast Forward ...** | | |
| **Merge 85** | `i` + `o` | `[space] B B C [space] R a d` **`io`** `[space] 4` |
| **Merge 102** | `B` + `B` | `[space]` **`BB`** `C [space] R a d io [space] 4` |
| **Merge 405** | `a` + `d` | `[space] BB C [space] R` **`ad`** `io[space] 4` |
| **Merge 800** | `BB` + `C` | `[space]` **`BBC`** `[space] R ad io [space] 4` |
| **Merge 1205** | `R` + `ad` | `[space] BBC[space]` **`Rad`** `io [space] 4` |
| **Merge 3100** | `[space]` + `BBC` | **` BBC`** `[space] Rad io[space] 4` |
| **Merge 3150** | `Rad` + `io` | ` BBC [space]` **`Radio`** `[space] 4` |
| **Merge 4200** | `[space]` + `Radio` | ` BBC` **` Radio`** `[space] 4` |

**Notice two critical behaviors:**
1. **Spaces are absorbed:** The tokenizer merges the space byte directly into the word. This is why modern tokenizers don't need arbitrary rule-based word splitting.
2. **Global non-linear building:** The algorithm didn't finish "BBC" before starting "Radio". It merged `i`+`o` at Step 85, left the rest alone, merged `B`+`B` at Step 102, and so on. It strictly follows global frequency distributions.

If the phrase `" BBC Radio"` occurs frequently enough, a later merge (e.g., Merge 15,000) might combine `[" BBC"]` + `[" Radio"]` into a single token: `[" BBC Radio"]`. This allows the LLM to process common compound phrases in a single embedding, massively increasing context window efficiency.

### Step C: Post-Processing (`TemplateProcessing`)
Once the model actually uses the tokenizer for training or inference, a post-processor applies special tokens automatically. 

If we pass `" BBC Radio 4"` into our finalized tokenizer, the pipeline executes:
1. **String Input:** `" BBC Radio 4"`
2. **BPE Application:** `[" BBC", " Radio", " 4"]`
3. **Template Wrapping:** `["<s>", " BBC", " Radio", " 4", "</s>"]`

This resulting sequence of integer token IDs is what is finally fed into the Llama architecture's embedding layer.
