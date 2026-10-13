# Reproduciblity Steps

## Requirements

Ensure you have python 3.8+ installed, then install the required dependencies:

```bash
pip install torch transformers datasets sentencepiece pyyaml matplotlib seaborn pandas numpy
```

## Phase 1 — Data Collection, Cleaning, Deduplication and Tokenization

### Data Collection
Extracts raw text from the provided Wikipedia dumps.
```bash
python3 wiki_extract.py
```

### Clean Hindi Corpus
Cleans and standardizes the extracted raw Hindi text.
```bash
python3 clean_corpus.py --config clean_dedup_config_hindi.yaml
```

### Clean Nepali Corpus
Cleans and standardizes the extracted raw Nepali text.
```bash
python3 clean_corpus.py --config clean_dedup_config_nepali.yaml
```

### Deduplicate Hindi Corpus
Removes duplicate documents from the Hindi training corpus.
```bash
python3 deduplicate_hindi.py --config clean_dedup_config_hindi.yaml
```

### Deduplicate Nepali Corpus
Removes duplicate documents from the Nepali training corpus.
```bash
python3 deduplicate_nepali.py --config clean_dedup_config_nepali.yaml
```

### Split Corpus
Splits the cleaned corpus into train, validation, and test sets.
```bash
python3 split_corpus.py
```

### Train Hindi Tokenizer
Trains the BPE tokenizer (v10000) on the Hindi corpus.
```bash
python3 -m hindi.tokenizer.tokenizer --config hindi/tokenizer/tokenizer_config_hindi.yaml
```

### Train Nepali Tokenizer
Trains the BPE tokenizer (v10000) on the Nepali corpus.
```bash
python3 -m nepali.tokenizer.tokenizer --config nepali/tokenizer/tokenizer_config_nepali.yaml
```

### Tokenize Hindi Splits
Tokenizes the raw Hindi corpus splits using the trained BPE model.
```bash
python3 -m hindi.tokenizer.tokenize_splits
```

### Tokenize Nepali Splits
Tokenizes the raw Nepali corpus splits using the trained BPE model.
```bash
python3 -m nepali.tokenizer.tokenize_splits
```

### Finalize Hindi 500M Training Set
Selects and formats exactly ~500M tokens for Hindi training.
```bash
python3 -m hindi.tokenizer.final_tokens_selector
```

### Finalize Nepali 500M Training Set
Selects and formats exactly ~500M tokens for Nepali training.
```bash
python3 -m nepali.tokenizer.final_tokens_selector
```

## Phase 2 — Model Implementation and Pretraining

### Pretrain Hindi Model
Trains the Hindi Phase 2 base model from scratch on the Hindi corpus.
```bash
python3 -m hindi.train.train
```

### Pretrain Nepali Model
Trains the Nepali Phase 2 base model from scratch on the Nepali corpus.
```bash
python3 -m nepali.train.train
```

### Resume Hindi Pretraining
Resumes Hindi training from a saved checkpoint.
```bash
python3 -m hindi.train.train --resume hindi/outputs/checkpoint.pt
```

### Evaluate Hindi Language Model
Evaluates perplexity and basic metrics of the Hindi base model.
```bash
python3 -m hindi.eval.run_evaluation
```

### Evaluate Nepali Language Model
Evaluates perplexity and basic metrics of the Nepali base model.
```bash
python3 -m nepali.eval.run_evaluation
```

### Evaluate Hindi Generation
Evaluates the generative capabilities of the Hindi base model.
```bash
python3 -m hindi.eval.run_generation_evaluation
```

### Evaluate Nepali Generation
Evaluates the generative capabilities of the Nepali base model.
```bash
python3 -m nepali.eval.run_generation_evaluation
```

### Phase 2 Attention Analysis (Hindi)
Analyzes attention head roles (local, long-range, focused) on the pretrained Hindi model.
```bash
python3 -m hindi.eval.attention_analysis
```

### Phase 2 Attention Analysis (Nepali)
Analyzes attention head roles (local, long-range, focused) on the pretrained Nepali model.
```bash
python3 -m nepali.eval.attention_analysis
```


## Phase 3 — Reasoning Dataset, SFT and Attention Analysis

### Generate Logical Dataset
Generates the underlying logical reasoning graphs and attributes.
```bash
python3 -m phase3.scripts.dataset.generate_logical_dataset
```

### Realize Text
Converts the logical reasoning structures into natural language text templates for both Hindi and Nepali.
```bash
python3 -m phase3.scripts.dataset.realize_text
```

### Compile Dataset
Compiles the final train, validation, and test splits into final JSONL datasets.
```bash
python3 -m phase3.scripts.dataset.compile_dataset
```

### Audit Dataset
Runs automated validation and leakage checks on the generated dataset splits.
```bash
python3 -m phase3.scripts.dataset.audit_dataset
```

### SFT Training (Hindi)
Fine-tunes the pretrained Hindi Phase 2 base model using LoRA on the reasoning dataset.
```bash
python3 -m phase3.scripts.training.sft_train --lang hi
```

### SFT Training (Nepali)
Fine-tunes the pretrained Nepali Phase 2 base model using LoRA on the reasoning dataset.
```bash
python3 -m phase3.scripts.training.sft_train --lang ne
```

### Evaluate Hindi Reasoning (Pretrained)
Evaluates exact match and reasoning correctness for the base Hindi model.
```bash
python3 -m phase3.scripts.evaluation.evaluate_reasoning --lang hi
```

### Evaluate Hindi Reasoning (SFT)
Evaluates exact match and reasoning correctness for the fine-tuned Hindi model.
```bash
python3 -m phase3.scripts.evaluation.evaluate_reasoning --lang hi --sft
```

### Evaluate Nepali Reasoning (Pretrained)
Evaluates exact match and reasoning correctness for the base Nepali model.
```bash
python3 -m phase3.scripts.evaluation.evaluate_reasoning --lang ne
```

### Evaluate Nepali Reasoning (SFT)
Evaluates exact match and reasoning correctness for the fine-tuned Nepali model.
```bash
python3 -m phase3.scripts.evaluation.evaluate_reasoning --lang ne --sft
```

### Aggregate Reasoning Results
Aggregates the evaluation results to produce final overall performance metrics.
```bash
python3 -m phase3.scripts.evaluation.aggregate_reasoning_results
```

### Error Analysis
Classifies the incorrect model predictions into semantic error categories.
```bash
python3 -m phase3.scripts.evaluation.error_analysis
```

### Phase 3 Attention Analysis (Hindi)
Measures post-finetuning attention distance and entropy, extracts head identities, and generates heatmaps.
```bash
python3 -m phase3.scripts.evaluation.attention_analysis --lang hi
python3 -m phase3.scripts.evaluation.attention_analysis --lang hi --sft
```

### Phase 3 Attention Analysis (Nepali)
Measures post-finetuning attention distance and entropy, extracts head identities, and generates heatmaps.
```bash
python3 -m phase3.scripts.evaluation.attention_analysis --lang ne
python3 -m phase3.scripts.evaluation.attention_analysis --lang ne --sft
```

## Bonus

### Pretrain No-RoPE Hindi Model
Trains a bonus Hindi model from scratch with the RoPE component disabled in the architecture.
```bash
python3 -m hindi.train.train --out-dir hindi_outputs_bonus
```

### Evaluate No-RoPE Hindi Model
Runs the Phase 2 evaluation on the ablated Hindi model to observe differences without positional embeddings.
```bash
python3 -m hindi.eval.run_evaluation --checkpoint hindi_outputs_bonus/best_model.pt
```


## Project Structure

```text
.
├── hindi/                     # Phase 1 & 2 Hindi Implementation
├── nepali/                    # Phase 1 & 2 Nepali Implementation
├── phase3/                    # Phase 3 Logical Reasoning and SFT Implementation
├── report/                    # Final rendered Phase 3 Markdown report
├── hindi_outputs_bonus/       # Output directory for the bonus No-RoPE ablation study
└── README.md                  # This file
```

## Google Drive

All checkpoints, final datasets, outputs, models, and metadata artifacts required for this project are stored in a single root folder. 

**Root Project Folder Link:**
https://drive.google.com/drive/folders/1VWXlchthvJWgXV0KNN6QsS4PD-7-ZvM7?usp=sharing

*(Note: The above root folder contains everything. There are no separate links for individual checkpoints.)*
