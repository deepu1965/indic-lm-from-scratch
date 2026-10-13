import argparse
import os
import json
import re
import time
import unicodedata
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import trafilatura

# Expanded Configuration with category and archive seeds for massive link discovery
LANG_CONFIG = {
    'nepali': {
        'script_range': ('\u0900', '\u097F'),
        'seeds': [
            'https://www.setopati.com/',
            'https://www.setopati.com/social',
            'https://www.setopati.com/kinmel',
            'https://www.setopati.com/opinion',
            'https://www.setopati.com/sports',
            'https://ekantipur.com/',
            'https://ekantipur.com/news',
            'https://ekantipur.com/business',
            'https://ekantipur.com/opinion',
            'https://www.onlinekhabar.com/',
            'https://www.onlinekhabar.com/content/news',
            'https://ratopati.com/',
            'https://ratopati.com/category/news',
            'https://annapurnapost.com/'
        ],
        'chars_per_token': 3.5
    }
}

def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_url(url, allowed_domains):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return any(domain.endswith(d) for d in allowed_domains) and parsed.scheme in ['http', 'https']
    except:
        return False

def get_script_ratio(text, script_range):
    if not text: return 0.0
    script_chars = sum(1 for c in text if script_range[0] <= c <= script_range[1])
    alpha_chars = sum(1 for c in text if c.isalpha())
    return script_chars / max(alpha_chars, 1)

def scrape_url(url, allowed_domains, session):
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return url, None, []
        
        # Extract text using trafilatura
        text = trafilatura.extract(resp.content, include_comments=False, include_tables=False)
        
        # Extract links using BeautifulSoup to fuel the crawler
        soup = BeautifulSoup(resp.content, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            absolute_url = urljoin(url, a['href']).split('#')[0]
            if is_valid_url(absolute_url, allowed_domains):
                links.add(absolute_url)
                
        return url, text, list(links)
    except Exception as e:
        return url, None, []

def main():
    parser = argparse.ArgumentParser(description="Incremental Web Crawler for NLP Corpus")
    parser.add_argument('--language', type=str, required=True, choices=['telugu', 'nepali'], help='Target language')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--target_tokens', type=int, default=110_000_000, help='Stop after reaching this many estimated tokens')
    parser.add_argument('--workers', type=int, default=10, help='Number of concurrent workers')
    
    args = parser.parse_args()
    
    config = LANG_CONFIG[args.language]
    output_dir = os.path.join(args.output_dir, args.language, 'manual_scrape')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'scraped_{args.language}.jsonl')
    
    allowed_domains = [urlparse(seed).netloc.replace('www.', '') for seed in config['seeds']]
    
    visited_urls = set()
    queue = config['seeds'].copy()
    
    total_est_tokens = 0
    total_articles = 0
    
    if os.path.exists(output_file):
        print(f"Found existing output file {output_file}. Resuming...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                visited_urls.add(record['source_name'])
                char_count = record.get('clean_char_count', len(record['text']))
                total_est_tokens += int(char_count / config['chars_per_token'])
                total_articles += 1
                
        print(f"Resumed state: {total_articles} articles, ~{total_est_tokens:,} tokens.")
        
        # When resuming, filter seeds and ensure queue is populated
        queue = [s for s in config['seeds'] if s not in visited_urls]
        if not queue:
            # If all seeds were visited, re-add seeds so crawler can extract fresh links from them
            print("Refreshing link discovery queue from seed pages...")
            queue = config['seeds'].copy()
            # Allow seed pages to be re-crawled for link discovery
            for s in config['seeds']:
                visited_urls.discard(s)
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    
    print(f"Starting crawler for {args.language}. Target: {args.target_tokens:,} tokens.")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while total_est_tokens < args.target_tokens:
                # If queue is empty, refresh seeds to discover deeper links
                if not queue:
                    print("\nQueue empty. Re-scanning seed categories for fresh article links...")
                    queue = config['seeds'].copy()
                    for s in config['seeds']:
                        visited_urls.discard(s)
                
                # Process in batches
                batch_urls = []
                while queue and len(batch_urls) < args.workers * 2:
                    url = queue.pop(0)
                    if url not in visited_urls:
                        visited_urls.add(url)
                        batch_urls.append(url)
                
                if not batch_urls:
                    continue
                
                futures = {executor.submit(scrape_url, url, allowed_domains, session): url for url in batch_urls}
                
                for future in as_completed(futures):
                    url, text, new_links = future.result()
                    
                    # Add unvisited new links to queue
                    for link in new_links:
                        if link not in visited_urls:
                            queue.append(link)
                            
                    # Process article text
                    if text:
                        text = normalize_text(text)
                        
                        if len(text) > 200 and get_script_ratio(text, config['script_range']) > 0.5:
                            char_count = len(text)
                            est_tokens = int(char_count / config['chars_per_token'])
                            
                            record = {
                                'text': text,
                                'source_name': url,
                                'source_type': 'manual_scrape',
                                'doc_id': f'scrape_{args.language}_{total_articles}',
                                'clean_char_count': char_count,
                                'est_tokens': est_tokens
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            f.flush()
                            
                            total_articles += 1
                            total_est_tokens += est_tokens
                            
                            if total_articles % 100 == 0:
                                print(f"Progress: {total_articles:,} articles | ~{total_est_tokens:,} tokens / {args.target_tokens:,} (Queue size: {len(queue):,})")
                                
                    if total_est_tokens >= args.target_tokens:
                        print("\nTarget token count reached!")
                        break

    print("-" * 40)
    print("Crawling Completed or Stopped.")
    print(f"Total Articles: {total_articles:,}")
    print(f"Total Tokens:   ~{total_est_tokens:,}")
    print(f"Output saved to: {output_file}")
    
if __name__ == "__main__":
    main()
