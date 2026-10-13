# Phase 1 — REPORT

## 1. Language Selection

Hindi was selected as a high resource language due to its vast speaker base and the availability of substantial, though often noisy, public web data. 

Nepali is chosen as the second language, representing a lower-resource but relatively more data among given options.


## 2. Dataset Collection / Sources

### Hindi

Dataset/Drive Link : https://drive.google.com/drive/folders/1AkmoB6FqP3Oci_a5Jyjqe2hXmOvXljzi?usp=sharing

| Source | Type |
| ------ | ---- |
| Sangraha Verified | Public/Downloaded |
| Wikipedia | Manual |
| Wikisource | Manual |

### Nepali

Dataset/Drive Link : https://drive.google.com/drive/folders/1W1loIHtex70Ex5B_gkqrPsh8Y5_-i4Gb?usp=sharing

| Source | Type |
| -------- | ------ |
| Sangraha Verified | Public/Downloaded |
| Wikipedia | Manual |
| Wikibooks | Manual |
| Ekantipur (news_articles) | Manual |

## 3. Data Cleaning

The actual pipeline implementation applies:

1. **Unicode Normalization**: Converts all text to NFC form.
2. **MediaWiki Markup Cleanup (Manual Only)**: Strips leftover image options (`thumb`, `px`, etc.) from Wikipedia/Wikisource dumps.
3. **Character/Script Filtering**: Retains only `devanagari` and `latin` scripts. Removes Unicode category `C` (controls) while deliberately preserving digits (`N`), punctuation (`P`), and symbols (`S` - e.g., math operators, currency).
4. **Length Filtering (Manual Only)**: Requires documents to meet minimum character and word counts.
5. **Line Structure Filtering (Manual Only)**: Discards documents dominated by short lines (e.g., lists or heavily parsed tables).
6. **Repetition Filtering (Manual Only)**: Filters repetitive documents based on repeated word/trigram/paragraph ratios.
7. **Latin Ratio Filtering (All Data)**: Removes individual sentences where >85% of letters are Latin, effectively filtering out English insertions while preserving technical terms and acronyms.

### Hindi Cleaning Statistics
* **Before Cleaning**: 
  - Sangraha (Public): 523,738,930 words | 2,687,998,967 chars
  - Manual (Wiki): 110,278,151 words | 604,192,614 chars
  - Overall: 634,017,081 words | 3,292,191,581 chars
* **After Cleaning**: 
  - Sangraha (Public): 519,613,401 words | 2,657,302,773 chars
  - Manual (Wiki): 90,741,311 words | 485,509,768 chars
  - Overall: 610,354,712 words | 3,142,812,541 chars
* **Reduction Analysis**: Cleaning reduced the overall word count by ~3.7%. The manual sources saw a larger reduction (~17.7% of words removed) primarily due to strict length, line-structure, and repetition filters applied exclusively to them.

### Nepali Cleaning Statistics
* **Before Cleaning**:
  - Sangraha (Public): 473,174,562 words | 3,051,709,954 chars
  - Manual: 67,978,429 words | 441,861,052 chars
  - Overall: 541,152,991 words | 3,493,571,006 chars
* **After Cleaning**:
  - Sangraha (Public): 472,289,294 words | 3,037,933,116 chars
  - Manual: 61,752,912 words | 401,929,547 chars
  - Overall: 534,042,206 words | 3,439,862,663 chars
* **Reduction Analysis**: Cleaning reduced the overall word count by ~1.3%, with manual sources observing a ~9.2% reduction.

## 4. Deduplication

1. **Exact Deduplication**: SHA-256 document hashing.
2. **Near Deduplication (MinHash + LSH)**: Detects near-duplicates using word n-gram shingles, MinHash permutations, and Locality Sensitive Hashing (LSH).

When duplicates are detected across source boundaries, the pipeline enforces strict source-priority rules to retain the highest-quality version.

### Hindi
* **Configuration**: `k=100`, `bands=20`, `rows=5`, `shingle size=3`, `threshold=0.80`
* **Source Priority**: `Manual (Wikimedia/Wikisource) > Sangraha`
* **Exact Dedup Statistics**:
  - Documents removed: 0 (0.00%)
* **Near Dedup Statistics**:
  - Manual removed: docs=18,719 (6.64%) | words=1,127,592 (1.24%) | chars=7,096,883 (1.46%)
  - Public removed: docs=3,626 (0.30%) | words=1,141,018 (0.22%) | chars=5,806,868 (0.22%)
  - Overall removed: docs=22,345 (1.48%) | words=2,268,610 (0.37%) | chars=12,903,751 (0.41%)

### Nepali
* **Configuration**: `k=100`, `bands=20`, `rows=5`, `shingle size=3`, `threshold=0.90`
* **Source Priority**: `Wikimedia > News_articles > Sangraha`
* **Exact Dedup Statistics**:
  - Documents removed: 0 (0.00%)
* **Near Dedup Statistics**:
  - Wikimedia removed: docs=233 (0.94%) | words=43,123 (0.73%) | chars=287,574 (0.72%)
  - Manual removed: docs=7,970 (5.62%) | words=3,794,390 (6.80%) | chars=25,327,570 (7.00%)
  - Sangraha removed: docs=390 (0.02%) | words=276,136 (0.06%) | chars=1,745,281 (0.06%)
  - Overall removed: docs=8,593 (0.49%) | words=4,113,649 (0.77%) | chars=27,360,425 (0.80%)

**Analysis**: In both languages, exact deduplication removed zero documents (suggesting exact duplicates were  simply absent). Near-deduplication proved much more effective. Notably, the manual datasets in both languages contained the highest proportion of near-duplicates (~6.6% in Hindi, ~5.6% in Nepali), demonstrating the necessity of the LSH pipeline for aggregated Wiki/news data. The source priority rule ensures that if a Wikipedia article is duplicated on a public web page, the Wikipedia version is retained. This is done becuase the manually collected data is less.

## 5. Final Corpus After Deduplication

| Statistic | Hindi | Nepali |
| --------- | ----- | ------ |
| **Documents** | 1,482,739 | 1,730,416 |
| **Words** | 608,086,102 | 529,928,557 |
| **Characters**| 3,129,908,790 | 3,412,502,238 |

### Source-wise Breakdown
**Hindi**
- Wikimedia: 103,241 docs
- Wikisource: 159,877 docs
- Sangraha: 1,219,621 docs
*Manual Fraction: 17.7% of documents (263,118 / 1,482,739)*

**Nepali**
- Wikimedia: 24,491 docs
- Manual: 133,898 docs
- Sangraha: 1,572,027 docs
*Manual Fraction: 9.15% of documents (158,389 / 1,730,416)*

## 6. Train / Validation / Test Split
The final deduplicated corpora are split into Train (98%), Validation (1%), and Test (1%) sets using a deterministic hash-based assignment (`hash(seed + doc_id) % 10000`) with a seed of 42. Crucially, splitting is **source-stratified**: the assignment is performed independently per source group to guarantee that minority data sources are proportionately represented in the validation and test sets.

### Hindi
* **Train (98.02%)**: 1,453,379 docs | 596,059,531 words | 3,068,148,482 chars
* **Validation (0.98%)**: 14,595 docs | 5,972,729 words | 30,670,699 chars
* **Test (1.00%)**: 14,765 docs | 6,053,842 words | 31,089,609 chars
* **Per-Source Stratification**:
  - Wikimedia: Train (97.94%), Val (1.02%), Test (1.04%)
  - Wikisource: Train (97.95%), Val (0.99%), Test (1.06%)
  - Sangraha: Train (98.04%), Val (0.98%), Test (0.98%)

### Nepali
* **Train (98.01%)**: 1,695,928 docs | 519,397,539 words | 3,344,674,333 chars
* **Validation (1.00%)**: 17,325 docs | 5,281,199 words | 34,016,066 chars
* **Test (0.99%)**: 17,163 docs | 5,249,819 words | 33,811,839 chars
* **Per-Source Stratification**:
  - Wikimedia: Train (98.11%), Val (0.99%), Test (0.90%)
  - Manual: Train (98.07%), Val (1.01%), Test (0.93%)
  - Sangraha: Train (98.00%), Val (1.00%), Test (1.00%)

## 7. Tokenizer Method
A completely independent Byte-Pair Encoding (BPE) tokenizer was trained from scratch for each language. 
- **Framework**: Hugging Face `tokenizers` library (`BpeTrainer`).
- **Streaming & Chunking**: To accommodate 8GB RAM constraints during training, the corpus was streamed lazily. Very long documents were chunked at whitespace boundaries (`max_chunk_chars: 50000`) so words were never split aggressively across chunks. 

## 8. Vocabulary Size Experiments
Three candidate vocabulary sizes (5,000, 7,500, and 10,000) were trained and evaluated on the **Validation** set to compare fragmentation behavior. 

*Fertility measures the average number of subword tokens produced per word. A lower fertility indicates that words remain more intact, while a higher fertility means words are fragmented into smaller pieces.*

### Hindi Evaluation (Validation Split)
| Metric | 5,000 Vocab | 7,500 Vocab | 10,000 Vocab |
| ------ | ----------- | ----------- | ------------ |
| **Fertility** | 1.5456 | 1.4343 | 1.3740 |
| **Average Chars/Token** | 3.3224 | 3.5802 | 3.7373 |
| **Unknown Rate** | 1.083249e-07 | 1.167291e-07 | 1.218520e-07 |

Google Drive LINK of models and reports: https://drive.google.com/drive/folders/1L_sRxoWcZ_C8lTGKok2j6taAcgSz1AGN?usp=sharing

### Nepali Evaluation (Validation Split)
| Metric | 5,000 Vocab | 7,500 Vocab | 10,000 Vocab |
| ------ | ----------- | ----------- | ------------ |
| **Fertility** | 1.6715 | 1.5293 | 1.4509 |
| **Average Chars/Token** | 3.8535 | 4.2118 | 4.4392 |
| **Unknown Rate** | 1.132845e-07 | 1.238171e-07 | 1.305028e-07 |

Google Drive LINK of models and reports: https://drive.google.com/drive/folders/18q_epWB__QlKed76ZYEmG0XIyFvvXfQW?usp=sharing

**Interpretation**: In both languages, as vocabulary size increases, fertility predictably decreases and the average characters per token increases. Larger vocabularies compress the text more efficiently at the cost of a larger embedding matrix. 


## 9. Final Vocabulary Selection

### Hindi
**Final vocabulary size: 10,000**

*Justification:* Among the three vocabulary sizes tested, 10,000 gives the lowest fertility (1.374), meaning that Hindi text is represented using fewer subword tokens on average. It also has the highest average of 3.74 characters per token, so the same text requires fewer tokens compared with the smaller vocabularies. The unknown-token rate is extremely low for all three sizes and changes only negligibly, so increasing the vocabulary to 10,000 does not introduce any meaningful disadvantage in terms of unknown tokens. Considering these results, the 10K vocabulary provides the best tokenization efficiency among the tested sizes, despite its slightly larger vocabulary and embedding cost.

### Nepali
**Final vocabulary size: 10,000**

*Justification:* For Nepali, the 10,000 vocabulary also gives the lowest fertility (1.4509) among the three candidates. It produces an average of 4.44 characters per token, which means fewer tokens are needed to represent the same text compared with the 5K and 7.5K vocabularies. This is particularly useful for a fixed context length because more of the original text can be represented within that context. The unknown-token rate remains extremely low across all three vocabulary sizes, so the larger vocabulary does not provide a significant disadvantage in this respect. Overall, 10K gives the best tokenization efficiency among the tested options and is therefore selected as the final Nepali vocabulary.

## 10. Tokenization Examples
Below are references to the generated tokenization examples.

* **Hindi 5K / 7.5K / 10K**: Please refer to `hindi/tokenizer/reports/tokenization_examples_v5000.txt`, `v7500.txt`, and `v10000.txt`.
* **Nepali 5K / 7.5K / 10K**: Please refer to `nepali/tokenizer/reports/tokenization_examples_v5000.txt`, `v7500.txt`, and `v10000.txt`.


## 11. Manual vs Downloaded Data Split
### Hindi
| Category | Documents | Words | Characters | Tokens (10K Vocab) | Fraction (Docs) | Fraction (Tokens) |
| -------- | --------- | ----- | ---------- | ------------------ | --------------- | ----------------- |
| Manual (Wiki/Wikisource) | 263,118 | 89,613,719 | 478,412,885 | 128,010,298 | 17.74% | 15.29% |
| Downloaded/Public (Sangraha) | 1,219,621 | 518,472,383 | 2,651,495,905 | 709,468,307 | 82.26% | 84.71% |
| **Total** | **1,482,739** | **608,086,102** | **3,129,908,790** | **837,478,605** | **100%** | **100%** |

### Nepali
| Category | Documents | Words | Characters | Tokens (10K Vocab) | Fraction (Docs) | Fraction (Tokens) |
| -------- | --------- | ----- | ---------- | ------------------ | --------------- | ----------------- |
| Manual (Wikimedia+News) | 158,389 | 57,915,399 | 376,314,403 | 84,770,770 | 9.15% | 11.03% |
| Downloaded/Public (Sangraha) | 1,572,027 | 472,013,158 | 3,036,187,835 | 683,949,323 | 90.85% | 88.97% |
| **Total** | **1,730,416** | **529,928,557** | **3,412,502,238** | **768,720,093** | **100%** | **100%** |


# Phase 2 - Changes to tokeniser (fixed the decoder issue)


## 8. Vocabulary Size Experiments
Three candidate vocabulary sizes (5,000, 7,500, and 10,000) were trained and evaluated on the **Validation** set to compare fragmentation behavior. 

### Hindi Evaluation (Validation Split)
| Metric | 5,000 Vocab | 7,500 Vocab | 10,000 Vocab |
| ------ | ----------- | ----------- | ------------ |
| **Fertility** | 2.5457 | 2.4340 | 2.3735 |
| **Average Chars/Token** | 2.0172 | 2.1098 | 2.1635 |
| **Unknown Rate** | 6.576907e-08 | 6.878788e-08 | 7.053905e-08 |

Google Drive LINK of models and reports: https://drive.google.com/drive/folders/1L_sRxoWcZ_C8lTGKok2j6taAcgSz1AGN?usp=sharing

Note: check final versions

### Nepali Evaluation (Validation Split)
| Metric | 5,000 Vocab | 7,500 Vocab | 10,000 Vocab |
| ------ | ----------- | ----------- | ------------ |
| **Fertility** | 2.6695 | 2.5268 | 2.4483 |
| **Average Chars/Token** | 2.4128 | 2.5490 | 2.6308 |
| **Unknown Rate** | 7.093239e-08 | 7.493621e-08 | 7.734090e-08 |

Google Drive LINK of models and reports: https://drive.google.com/drive/folders/18q_epWB__QlKed76ZYEmG0XIyFvvXfQW?usp=sharing

Note: check final versions

**Interpretation**: In both languages, as vocabulary size increases, fertility predictably decreases and the average characters per token increases. Larger vocabularies compress the text more efficiently at the cost of a larger embedding matrix. 


## 9. Final Vocabulary Selection

### Hindi
**Final vocabulary size: 10,000**

*Justification:* Among the three vocabulary sizes tested, 10,000 gives the lowest fertility (2.3735), meaning that Hindi text is represented using fewer subword tokens on average. It also has the highest average of 2.16 characters per token, so the same text requires fewer tokens compared with the smaller vocabularies. The unknown-token rate is extremely low for all three sizes and changes only negligibly, so increasing the vocabulary to 10,000 does not introduce any meaningful disadvantage in terms of unknown tokens. Considering these results, the 10K vocabulary provides the best tokenization efficiency among the tested sizes, despite its slightly larger vocabulary and embedding cost.

### Nepali
**Final vocabulary size: 10,000**

*Justification:* For Nepali, the 10,000 vocabulary also gives the lowest fertility (2.4483) among the three candidates. It produces an average of 2.63 characters per token, which means fewer tokens are needed to represent the same text compared with the 5K and 7.5K vocabularies. This is particularly useful for a fixed context length because more of the original text can be represented within that context. The unknown-token rate remains extremely low across all three vocabulary sizes, so the larger vocabulary does not provide a significant disadvantage in this respect. Overall, 10K gives the best tokenization efficiency among the tested options and is therefore selected as the final Nepali vocabulary.

## 11. Manual vs Downloaded Data Split
### Hindi
| Category | Documents | Words | Characters | Tokens (10K Vocab) | Fraction (Docs) | Fraction (Tokens) |
| -------- | --------- | ----- | ---------- | ------------------ | --------------- | ----------------- |
| Manual (Wiki/Wikisource) | 263,118 | 89,613,719 | 478,412,885 | 221,129,136 | 17.74% | 15.29% |
| Downloaded/Public (Sangraha) | 1,219,621 | 518,472,383 | 2,651,495,905 | 1,225,558,541 | 82.26% | 84.71% |
| **Total** | **1,482,739** | **608,086,102** | **3,129,908,790** | **1,446,687,677** | **100%** | **100%** |

### Nepali
| Category | Documents | Words | Characters | Tokens (10K Vocab) | Fraction (Docs) | Fraction (Tokens) |
| -------- | --------- | ----- | ---------- | ------------------ | --------------- | ----------------- |
| Manual (Wikimedia+News) | 158,389 | 57,915,399 | 376,314,403 | 143,041,814 | 9.15% | 11.03% |
| Downloaded/Public (Sangraha) | 1,572,027 | 472,013,158 | 3,036,187,835 | 1,154,092,989 | 90.85% | 88.97% |
| **Total** | **1,730,416** | **529,928,557** | **3,412,502,238** | **1,297,134,803** | **100%** | **100%** |

