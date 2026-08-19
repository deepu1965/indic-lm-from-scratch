import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder

# We will test these vocabulary sizes as requested by the professor
VOCAB_SIZES = [5000, 8000, 10000]

def get_jsonl_files(language):
    """Find the final cleaned .jsonl corpus file."""
    final_file = f"./data/{language}/final_corpus/FINAL_{language}_corpus.jsonl"
    if os.path.exists(final_file):
        return [final_file]
    
    print(f"WARNING: Final corpus not found for {language}! Make sure you ran full_cleaning_pipeline.py first.")
    return []

def text_iterator(files, is_train=True, split_ratio=10):
    """
    Yields text from the JSONL files efficiently.
    Uses a deterministic split (e.g. every 10th document is for evaluation)
    to ensure Train/Eval sets do not overlap.
    """
    for file_path in files:
        if is_train:
            print(f"Reading {file_path} for TRAINING...")
        else:
            print(f"Reading {file_path} for EVALUATION...")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    record = json.loads(line)
                    text = record.get('text', '').strip()
                    if text:
                        # 10% evaluation, 90% training
                        is_eval = (line_idx % split_ratio == 0)
                        if is_train and not is_eval:
                            yield text
                        elif not is_train and is_eval:
                            yield text
                except json.JSONDecodeError:
                    pass
                except GeneratorExit:
                    return

def compute_metrics(tokenizer, files, max_eval_docs=50000):
    """
    Evaluates the tokenizer on a strictly held-out evaluation corpus.
    Calculates Fertility, UNK Rate, and Characters per Token.
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_unks = 0
    
    count = 0
    unk_token_id = tokenizer.token_to_id("[UNK]")
    
    for text in text_iterator(files, is_train=False):
        if count >= max_eval_docs:  # Stop after processing 'max_eval_docs' records
            break
            
        # FIX 1: Split on whitespace instead of regex for strict fertility baseline
        words = len(text.split())
        if words == 0:
            continue
            
        encoded = tokenizer.encode(text)
        
        total_words += words
        total_tokens += len(encoded.tokens)
        total_chars += len(text)
        
        # Count UNK tokens
        if unk_token_id is not None:
            total_unks += encoded.ids.count(unk_token_id)
        
        count += 1
        
    fertility = total_tokens / total_words if total_words > 0 else 0
    unk_rate = (total_unks / total_tokens) * 100 if total_tokens > 0 else 0
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    
    return fertility, unk_rate, chars_per_token

def train_for_language(language):
    files = get_jsonl_files(language)
    if not files:
        return
        
    print(f"\n======================================")
    print(f"Training Tokenizers for {language.upper()}")
    print(f"======================================")
    
    out_dir = f"./models/tokenizers/{language}"
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for vocab_size in VOCAB_SIZES:
        print(f"\n---> Training BPE Tokenizer with Vocab Size: {vocab_size}")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        # Add Pre-tokenizer and Decoder
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = BPEDecoder()  # FIX 8: Explicit decoder
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        # Train ONLY on the training split (90% of data)
        tokenizer.train_from_iterator(text_iterator(files, is_train=True), trainer=trainer)
        
        # Save model
        save_path = os.path.join(out_dir, f"bpe_{vocab_size}.json")
        tokenizer.save(save_path)
        print(f"Saved tokenizer to {save_path}")
        
        # Evaluate on the EVALUATION split (10% of data)
        print("Evaluating Metrics on Held-out Eval Set...")
        fertility, unk_rate, chars_per_token = compute_metrics(tokenizer, files)
        
        print(f"   Fertility Rate   : {fertility:.4f} tokens/word")
        print(f"   UNK Rate         : {unk_rate:.4f}%")
        print(f"   Characters/Token : {chars_per_token:.4f}")
        
        results[vocab_size] = {
            'fertility': fertility,
            'unk_rate': unk_rate,
            'chars_per_token': chars_per_token
        }
        
    print(f"\n==========================================")
    print(f"Final Comparison Table for {language.upper()}")
    print(f"==========================================")
    print(f"{'Vocab Size':<12} | {'Fertility':<10} | {'UNK Rate %':<10} | {'Chars/Token':<12}")
    print("-" * 52)
    for size in VOCAB_SIZES:
        r = results[size]
        print(f"{size:<12} | {r['fertility']:<10.4f} | {r['unk_rate']:<10.4f} | {r['chars_per_token']:<12.4f}")
    print("=" * 52)
    print("\n[NOTE] Do not blindly pick the lowest fertility. Compare the metrics above.")
    print("Choose the smallest vocabulary size that achieves near-best performance.")

def main():
    train_for_language("telugu")
    train_for_language("nepali")
    
if __name__ == "__main__":
    main()
