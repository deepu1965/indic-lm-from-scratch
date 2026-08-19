import json
import os
import re
import hashlib
from typing import Set
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def get_shingles(text: str, n: int = 5) -> Set[str]:
    # Basic word split for shingles
    words = re.findall(r'\w+', text.lower())
    shingles = set()
    if len(words) < n:
        if words:
            shingles.add(" ".join(words))
    else:
        for i in range(len(words) - n + 1):
            shingles.add(" ".join(words[i:i+n]))
    return shingles

import argparse

def main():
    parser = argparse.ArgumentParser(description="Deduplicate manual scrape data")
    parser.add_argument('--language', type=str, required=True, choices=['telugu', 'nepali'])
    args = parser.parse_args()
    
    language = args.language
    input_file = f"./data/{language}/manual_scrape/scraped_{language}.jsonl"
    output_dir = f"./data/{language}/cleaned"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"scraped_{language}_dedup.jsonl")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
        
    print(f"Starting Exact + Near Deduplication on {language.capitalize()} manual scrape data...")
    print(f"Input: {input_file}")
    
    # --- Exact Dedup State ---
    seen_hashes = set()
    whitespace_pattern = re.compile(r'\s+')
    
    # --- Near Dedup State (MinHash LSH) ---
    # Jaccard threshold of 0.8 is standard for near-duplicate text
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    seen_signatures = {}
    
    exact_duplicates = 0
    near_duplicates = 0
    total_processed = 0
    total_kept = 0
    
    print("Reading data...")
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines, desc="Deduplicating"):
            total_processed += 1
            record = json.loads(line)
            text = record.get('text', '')
            doc_id = record.get('doc_id', f'doc_{total_processed}')
            
            if not text.strip():
                continue
                
            # 1. Exact Deduplication
            normalized_text = whitespace_pattern.sub(' ', text).strip().lower()
            doc_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
            if doc_hash in seen_hashes:
                exact_duplicates += 1
                continue
            seen_hashes.add(doc_hash)
            
            # 2. Near Deduplication
            shingles = get_shingles(text)
            if not shingles:
                continue
                
            m = MinHash(num_perm=128)
            for d in shingles:
                m.update(d.encode('utf8'))
                
            candidates = lsh.query(m)
            is_near_dup = False
            if candidates:
                for cand_id in candidates:
                    cand_minhash = seen_signatures.get(cand_id)
                    if cand_minhash and m.jaccard(cand_minhash) >= 0.8:
                        is_near_dup = True
                        break
            
            if is_near_dup:
                near_duplicates += 1
                continue
                
            # If unique, insert into LSH and save
            lsh.insert(doc_id, m)
            seen_signatures[doc_id] = m
            
            f_out.write(line)
            total_kept += 1

    print("\n" + "="*40)
    print("Deduplication Results:")
    print("="*40)
    print(f"Total articles processed:   {total_processed:,}")
    print(f"Exact duplicates dropped:   {exact_duplicates:,}")
    print(f"Near duplicates dropped:    {near_duplicates:,}")
    print(f"Total clean articles kept:  {total_kept:,}")
    print(f"Output saved to:            {output_file}")
    
    # Calculate approximate tokens remaining
    chars_per_token = 5.0 if language == 'telugu' else 3.5
    with open(output_file, 'r', encoding='utf-8') as f:
        remaining_tokens = sum(json.loads(line).get('est_tokens', 0) for line in f)
    print(f"Estimated tokens remaining: ~{remaining_tokens:,}")

if __name__ == "__main__":
    main()
