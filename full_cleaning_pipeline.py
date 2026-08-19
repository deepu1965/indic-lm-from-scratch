import os
import json
import re
import zipfile
import unicodedata
import hashlib
from typing import Set
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

class CorpusCleaner:
    def __init__(self, language):
        self.language = language
        self.seen_hashes = set()
        self.lsh = MinHashLSH(threshold=0.8, num_perm=128)
        self.seen_signatures = {}
        
        # Language script ranges for filtering
        self.script_ranges = {
            'telugu': (0x0C00, 0x0C7F),
            'nepali': (0x0900, 0x097F) # Devanagari
        }
        
        # Metrics
        self.metrics = {
            'raw_documents': 0,
            'docs_removed_script': 0,
            'docs_removed_short': 0,
            'docs_removed_exact': 0,
            'docs_removed_near': 0,
            'final_documents': 0,
            'raw_chars': 0,
            'final_chars': 0,
            'final_tokens': 0
        }
        
    def _is_valid_script(self, text):
        """Ensure at least 50% of the alphabetic characters belong to the target language script."""
        if not text:
            return False
        
        start, end = self.script_ranges[self.language]
        lang_chars = 0
        total_alpha = 0
        
        for char in text:
            if char.isalpha():
                total_alpha += 1
                if start <= ord(char) <= end:
                    lang_chars += 1
                    
        if total_alpha == 0:
            return False
            
        return (lang_chars / total_alpha) > 0.5

    def _get_shingles(self, text, n=5):
        words = re.findall(r'\w+', text.lower())
        shingles = set()
        if len(words) < n:
            if words:
                shingles.add(" ".join(words))
        else:
            for i in range(len(words) - n + 1):
                shingles.add(" ".join(words[i:i+n]))
        return shingles

    def clean_text(self, text):
        # 1. Unicode Normalization
        text = unicodedata.normalize('NFC', text)
        
        # 2. HTML/Boilerplate removal (Strip standard HTML tags if any slipped through)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 3. URL/Email cleanup
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Invalid character filtering (remove obscure control characters except newline)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == '\n')
        
        # 5. Whitespace normalization
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def _process_text(self, raw_text, record, out_f, input_file):
        self.metrics['raw_documents'] += 1
        self.metrics['raw_chars'] += len(raw_text)
        
        # Apply text cleaning
        clean_txt = self.clean_text(raw_text)
        
        # Very-short document filtering
        if len(clean_txt) < 100:
            self.metrics['docs_removed_short'] += 1
            return
            
        # Language/script filtering
        if not self._is_valid_script(clean_txt):
            self.metrics['docs_removed_script'] += 1
            return
            
        # Exact deduplication
        normalized_lower = re.sub(r'\s+', ' ', clean_txt).lower()
        doc_hash = hashlib.sha256(normalized_lower.encode('utf-8')).hexdigest()
        if doc_hash in self.seen_hashes:
            self.metrics['docs_removed_exact'] += 1
            return
        self.seen_hashes.add(doc_hash)
        
        # Near-duplicate filtering (MinHash)
        shingles = self._get_shingles(clean_txt)
        if not shingles:
            self.metrics['docs_removed_short'] += 1
            return
            
        m = MinHash(num_perm=128)
        for d in shingles:
            m.update(d.encode('utf8'))
            
        candidates = self.lsh.query(m)
        is_near_dup = False
        for cand_id in candidates:
            cand_minhash = self.seen_signatures.get(cand_id)
            if cand_minhash and m.jaccard(cand_minhash) >= 0.8:
                is_near_dup = True
                break
                
        if is_near_dup:
            self.metrics['docs_removed_near'] += 1
            return
            
        doc_id = record.get('doc_id', f"doc_{self.metrics['raw_documents']}")
        self.lsh.insert(doc_id, m)
        self.seen_signatures[doc_id] = m
        
        # Passed all filters!
        chars_per_tok = 5.0 if self.language == 'telugu' else 3.5
        est_toks = int(len(clean_txt) / chars_per_tok)
        
        final_record = {
            'text': clean_txt,
            'doc_id': doc_id,
            'source': record.get('source_type', 'unknown')
        }
        
        out_f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
        
        self.metrics['final_documents'] += 1
        self.metrics['final_chars'] += len(clean_txt)
        self.metrics['final_tokens'] += est_toks

    def process_file(self, input_file, out_f):
        if input_file.endswith('.jsonl'):
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {os.path.basename(input_file)}"):
                    try:
                        record = json.loads(line)
                        raw_text = record.get('text', '')
                        if raw_text:
                            self._process_text(raw_text, record, out_f, input_file)
                    except:
                        continue
        elif input_file.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(input_file)
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            for idx, text in tqdm(enumerate(df[text_col]), total=len(df), desc=f"Processing {os.path.basename(input_file)}"):
                if not text:
                    continue
                record = {'source_type': 'parquet_download', 'doc_id': f"{os.path.basename(input_file)}_{idx}"}
                self._process_text(str(text), record, out_f, input_file)

def get_data_files(language):
    files = []
    base_dir = f"./data/{language}"
    # Prevent reading from final_corpus if it already exists from a previous run
    for root, _, filenames in os.walk(base_dir):
        if 'final_corpus' in root or 'manual_scrape' in root:
            continue
        for f in filenames:
            if f.endswith('.jsonl') or f.endswith('.parquet'):
                files.append(os.path.join(root, f))
    return files

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True, choices=['telugu', 'nepali'])
    args = parser.parse_args()
    
    language = args.language
    input_files = get_data_files(language)
    
    out_dir = f"./data/{language}/final_corpus"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"FINAL_{language}_corpus.jsonl")
    
    print(f"=== Starting Full Cleaning Pipeline for {language.upper()} ===")
    cleaner = CorpusCleaner(language)
    
    with open(out_file, 'w', encoding='utf-8') as out_f:
        for f in input_files:
            cleaner.process_file(f, out_f)
            
    print("\n" + "="*40)
    print(f"Final Pipeline Metrics for {language.upper()}")
    print("="*40)
    for k, v in cleaner.metrics.items():
        print(f"{k.ljust(25)}: {v:,}")
    print("="*40)
    print(f"Saved highly cleaned corpus to: {out_file}")

if __name__ == "__main__":
    main()
