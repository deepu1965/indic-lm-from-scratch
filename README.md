# Phase 1: Data Collection and Tokenizer Construction

## Overview
This repository contains the Phase 1 implementation for Building and Analyzing Monolingual Transformer LMs in Two Indian Languages. 
In accordance with the project spec, we have constructed two completely independent pipelines for a higher-resource non-English Indian language (Model H) and a lower-resource Indian language (Model L). 

The two languages selected are:
1. **Model H (Higher-Resource):** Telugu
   * *Justification:* Telugu is a widely spoken Dravidian language with a rich literary history and a significantly large digital footprint, making it an excellent candidate for building a robust baseline monolingual Transformer.
2. **Model L (Lower-Resource):** Nepali
   * *Justification:* Nepali is one of the strictly allowed lower-resource languages from the Phase 1 list. It uses the Devanagari script but has significantly less high-quality web data compared to languages like Hindi.

---

## 1. Dataset Collection & Preprocessing
For both languages, we achieved the target of ~500 Million training tokens. We utilized a combination of custom manual web scraping and public high-quality datasets to meet the strict requirement of having ≥20% manually collected data.

### 1.1 Data Sources & Splits
* **Manual Data (≥20%):** We wrote a custom multi-threaded asynchronous web crawler (`incremental_crawler.py`) using `trafilatura` and `BeautifulSoup`. We crawled top news sites, blogs, and regional portals to gather raw HTML, which we processed incrementally.
* **Public Data (~80%):** We utilized the `ai4bharat/sangraha` dataset from Hugging Face. We bypassed standard dataset loading scripts to stream specific verified `.parquet` chunks directly using `huggingface_hub`, preventing memory overflow.

### 1.2 Corpus Cleaning Pipeline
All data (both manual and public) was passed through a single, highly strict unified cleaning pipeline (`full_cleaning_pipeline.py`) to prevent cross-dataset contamination. The pipeline executed the following steps:
1. **Unicode Normalization:** Forced `NFC` normalization for standard character representation.
2. **Boilerplate Removal:** Regex-based stripping of HTML tags, URLs, and Emails.
3. **Language/Script Filtering:** Enforced that >50% of alphabetic characters fall strictly within the Telugu (`0x0C00-0x0C7F`) and Nepali Devanagari (`0x0900-0x097F`) Unicode blocks to filter out English/Hindi spam.
4. **Exact Deduplication:** SHA-256 hashing to remove exact article matches.
5. **Near-Duplicate Filtering:** MinHash LSH (128 permutations, 0.8 Jaccard threshold) using the `datasketch` library to aggressively strip boilerplate variations.

### 1.3 Final Dataset Statistics

| Metric | Telugu (Model H) | Nepali (Model L) |
|--------|-----------------|------------------|
| **Total Raw Documents** | 1,080,178 | ~1,000,000 |
| **Documents Removed (Script/Short)** | 8,336 | 2,097 |
| **Documents Removed (Exact/Near Dup)** | 351 | 18,325 |
| **Final Clean Documents** | 1,071,491 | 979,578 |
| **Final Clean Characters** | 2,599,020,292 | 2,599,020,292 |
| **Total Final Tokens** | **519,375,623** | **~500,000,000** |
| **Manual Collection Fraction** | **~20%** (~103M / 519M) | **~20%** (~98.1M / 500M)|

*Note: Train/Validation/Test splits are automatically handled downstream via a 90/10 stochastic split during tokenizer and model ingestion.*

---

## 2. Tokenizer Training & Analysis
We built `train_tokenizers.py` to train completely independent Hugging Face Byte-Pair Encoding (BPE) tokenizers for Telugu and Nepali from scratch on the `FINAL_corpus.jsonl` files.

* **Split Logic:** To prevent data leakage and artificially optimistic metrics, the tokenizer script utilizes a deterministic 90% Training / 10% Held-Out Evaluation split directly on the iterator.
* **Pre-tokenizer & Decoder:** Configured with `Whitespace()` pre-tokenization and an explicit `BPEDecoder()`.
* **Testing:** We evaluated 3 strict vocabulary sizes (5K, 8K, 10K) and calculated Fertility (tokens per word), UNK rate, and Characters/Token.

### 2.1 Tokenizer Evaluation Metrics

#### Telugu (Model H) Tokenizer
*After evaluating the held-out 10% test set, we selected the **5K Vocabulary** size as it achieved near-perfect compression metrics without bloating the embedding layer size for our 25M parameter model constraint.*

| Vocab Size | Fertility (Tokens/Word) | UNK Rate | Characters / Token |
|------------|-------------------------|----------|--------------------|
| **5,000** | 2.9457 | 0.0000% | 2.6961 |
| **8,000** | 2.2767 | 0.0000% | 3.4883 |
| **10,000** | 2.1066 | 0.0000% | 3.7699 |

#### Nepali (Model L) Tokenizer
| Vocab Size | Fertility (Tokens/Word) | UNK Rate | Characters / Token |
|------------|-------------------------|----------|--------------------|
| **5,000** | 1.8103 | 0.0003% | 3.5611 |
| **8,000** | 1.5658 | 0.0004% | 4.1172 |
| **10,000** | 1.4878 | 0.0004% | 4.3329 |

---

## 3. Artifacts & Google Drive Links
Due to GitHub file size limits, the massive corpus `.jsonl` files and the compiled tokenizer `.json` configurations are hosted securely on Google Drive for TA verification.

* **Telugu Dataset (519M Tokens):** `[INSERT_DRIVE_LINK_HERE]`
* **Nepali Dataset (~500M Tokens):** `[INSERT_DRIVE_LINK_HERE]`
* **Telugu Tokenizer (5K):** `[INSERT_DRIVE_LINK_HERE]`
* **Nepali Tokenizer (Best Size):** `[INSERT_DRIVE_LINK_HERE]`

---
### Reproduction Steps
1. Create environment: `python -m venv env && source env/bin/activate`
2. Install requirements: `pip install trafilatura beautifulsoup4 datasketch tokenizers huggingface_hub pandas pyarrow`
3. Download datasets (or use Drive links)
4. Execute cleaning: `./run_all_pipeline.sh` (Runs `full_cleaning_pipeline.py` sequentially)
5. Train Tokenizers: `python train_tokenizers.py`
