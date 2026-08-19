import os
import json
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
HF_TOKEN = os.environ.get("HF_TOKEN")
# Sangraha (IndicCorp v2) verified folder codes
LANG_CONFIG = {
    'telugu': {
        'sangraha_code': 'tel',
        'chars_per_token': 5.0,
        'target_tokens': 450_000_000
    },
    'nepali': {
        'sangraha_code': 'nep',
        'chars_per_token': 3.5,
        'target_tokens': 450_000_000
    }
}

def main():
    print("=== ai4bharat/sangraha Downloader ===")
    
    if not HF_TOKEN:
        print("Please paste your Hugging Face token in the HF_TOKEN variable at the top of the script. Exiting.")
        return
        
    start_idx = 1 # Skip the 0th file as requested
    repo_id = "ai4bharat/sangraha"
    
    print(f"\nConnecting to Hugging Face Hub (Repo: {repo_id})...")
    api = HfApi(token=HF_TOKEN)
    
    try:
        all_files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Failed to access {repo_id}. Did you accept the dataset conditions on Hugging Face?")
        print(f"Error: {e}")
        return

    # Process both languages
    for language, config in LANG_CONFIG.items():
        print(f"\n" + "="*40)
        print(f"Processing {language.upper()}...")
        print("="*40)
        
        output_dir = f"./data/{language}/downloaded"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"public_{language}_sangraha.jsonl")
        
        # Filter files for the specific language
        sangraha_code = config['sangraha_code']
        lang_files = [f for f in all_files if f.startswith(f"verified/{sangraha_code}/") and f.endswith(".parquet")]
        lang_files.sort()
        
        if not lang_files:
            print(f"Could not find any verified parquets for {sangraha_code}. Skipping.")
            continue
            
        print(f"Found {len(lang_files)} verified parquet files for {language}.")
        
        if start_idx < len(lang_files):
            files_to_process = lang_files[start_idx:]
            print(f"Starting from file index {start_idx} (Skipping 0th file)")
        else:
            print(f"Start index {start_idx} is larger than available files for {language}. Skipping.")
            continue

        total_est_tokens = 0
        doc_count = 0
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for file_path in files_to_process:
                if total_est_tokens >= config['target_tokens']:
                    print(f"\nTarget of {config['target_tokens']:,} tokens reached for {language}!")
                    break
                    
                print(f"\nDownloading: {file_path}")
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id, 
                        filename=file_path, 
                        repo_type="dataset", 
                        token=HF_TOKEN
                    )
                    
                    print(f"Reading parquet file...")
                    df = pd.read_parquet(local_path)
                    
                    text_col = 'text' if 'text' in df.columns else df.columns[0]
                    
                    for text in df[text_col]:
                        if not text or len(str(text)) < 100:
                            continue
                            
                        clean_text = " ".join(str(text).split())
                        char_count = len(clean_text)
                        est_tokens = int(char_count / config['chars_per_token'])
                        
                        record = {
                            'text': clean_text,
                            'source_name': f"{repo_id}/{file_path}",
                            'source_type': 'downloaded',
                            'doc_id': f'dl_sangraha_{language}_{doc_count}',
                            'clean_char_count': char_count,
                            'est_tokens': est_tokens
                        }
                        
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        
                        doc_count += 1
                        total_est_tokens += est_tokens
                        
                        if doc_count % 10000 == 0:
                            print(f"Progress ({language}): {total_est_tokens:,} tokens / {config['target_tokens']:,} target")
                            f.flush()
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        print(f"\nCompleted {language.upper()}.")
        print(f"Total Documents: {doc_count:,}")
        print(f"Total Tokens:    ~{total_est_tokens:,}")
        print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
