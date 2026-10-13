import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
import time
import itertools
from pathlib import Path

from nepali.model import DecoderOnlyTransformer
from nepali.model.config import ModelConfig


def get_special_token_id(tokenizer_path, token_str):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for added in data.get("added_tokens", []):
        if added.get("content") == token_str:
            return added["id"]
    vocab = data.get("model", {}).get("vocab", {})
    if token_str in vocab:
        return vocab[token_str]
    raise ValueError(f"Token {token_str} not found in {tokenizer_path}")


def evaluate_intrinsic(
    model, 
    context_length, 
    raw_path, 
    tok_path, 
    bos_id, 
    eos_id, 
    device, 
    smoke_test=False
):
    total_bytes = 0
    total_words = 0
    total_raw_tokens = 0  # Excluding BOS/EOS
    
    total_nll = 0.0
    total_predicted_tokens = 0
    
    buffer = []
    
    print(f"Evaluating raw: {raw_path}")
    print(f"Evaluating tok: {tok_path}")
    
    start_time = time.time()
    
    with open(raw_path, "r", encoding="utf-8") as f_raw, open(tok_path, "r", encoding="utf-8") as f_tok:
        for idx, (raw_line, tok_line) in enumerate(itertools.zip_longest(f_raw, f_tok)):
            if smoke_test and idx >= 100:
                break
                
            if raw_line is None or tok_line is None:
                raise ValueError(f"File length mismatch at line {idx}! Raw has line: {raw_line is not None}, Tok has line: {tok_line is not None}")
                
            raw_doc = json.loads(raw_line)
            tok_doc = json.loads(tok_line)
            
            # Alignment check
            if "id" in raw_doc and "id" in tok_doc:
                if raw_doc['id'] != tok_doc['id']:
                    raise ValueError(f"ID mismatch at line {idx}: raw={raw_doc['id']}, tok={tok_doc['id']}")
            
            text = raw_doc.get("text", "")
            tokens = tok_doc.get("tokens", [])
            
            # Fertility metrics
            total_bytes += len(text.encode("utf-8"))
            total_words += len(text.split())
            total_raw_tokens += len(tokens)
            
            # Pack tokens for evaluation
            buffer.append(bos_id)
            buffer.extend(tokens)
            buffer.append(eos_id)
            
            # Process full chunks
            while len(buffer) >= context_length + 1:
                chunk = buffer[:context_length + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                labels = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
                
                with torch.inference_mode():
                    logits = model(input_ids)
                
                mask = (labels != bos_id) & (labels != eos_id)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
                
                total_nll += (loss * mask.view(-1)).sum().item()
                total_predicted_tokens += mask.sum().item()
                
                del buffer[:context_length]
    
    # Process remainder of the buffer
    if len(buffer) > 1:
        input_ids = torch.tensor(buffer[:-1], dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.tensor(buffer[1:], dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.inference_mode():
            logits = model(input_ids)
            
        mask = (labels != bos_id) & (labels != eos_id)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        
        total_nll += (loss * mask.view(-1)).sum().item()
        total_predicted_tokens += mask.sum().item()

    if total_predicted_tokens == 0 or total_bytes == 0 or total_words == 0 or total_raw_tokens == 0:
        raise ValueError("One of the metric denominators is zero. Check input files.")

    cross_entropy_loss = total_nll / total_predicted_tokens
    perplexity = math.exp(cross_entropy_loss)
    bpb = total_nll / (math.log(2) * total_bytes)
    
    tokens_per_word = total_raw_tokens / total_words
    bytes_per_token = total_bytes / total_raw_tokens
    
    elapsed = time.time() - start_time
    print(f"Evaluation finished in {elapsed:.2f}s")
    
    return {
        "cross_entropy_loss": cross_entropy_loss,
        "perplexity": perplexity,
        "bits_per_byte": bpb,
        "tokens_per_word": tokens_per_word,
        "bytes_per_token": bytes_per_token,
        "total_predicted_tokens": total_predicted_tokens,
        "total_tokens": total_raw_tokens,
        "total_words": total_words,
        "total_bytes": total_bytes
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--model-cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--raw-data", type=str, required=True)
    parser.add_argument("--tokenized-data", type=str, required=True)
    parser.add_argument("--tokenizer-json", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    bos_id = get_special_token_id(args.tokenizer_json, "[BOS]")
    eos_id = get_special_token_id(args.tokenizer_json, "[EOS]")
    print(f"Loaded BOS ID: {bos_id}, EOS ID: {eos_id}")
    
    model_cfg = ModelConfig.from_yaml(args.model_cfg)
    model = DecoderOnlyTransformer(model_cfg)
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    metrics = evaluate_intrinsic(
        model=model,
        context_length=model_cfg.context_length,
        raw_path=args.raw_data,
        tok_path=args.tokenized_data,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        smoke_test=args.smoke_test
    )
    
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    csv_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", encoding="utf-8") as f:
        if not csv_exists:
            f.write("language,split,cross_entropy_loss,perplexity,bits_per_byte,tokens_per_word,bytes_per_token,total_predicted_tokens,total_tokens,total_words,total_bytes\n")
        f.write(f"nepali,{args.split},{metrics['cross_entropy_loss']},{metrics['perplexity']},{metrics['bits_per_byte']},{metrics['tokens_per_word']},{metrics['bytes_per_token']},{metrics['total_predicted_tokens']},{metrics['total_tokens']},{metrics['total_words']},{metrics['total_bytes']}\n")
        
    print(f"Metrics saved to {args.out_json} and {args.out_csv}")
    print(f"Cross Entropy: {metrics['cross_entropy_loss']:.4f}")
    print(f"Perplexity:    {metrics['perplexity']:.4f}")
    print(f"BPB:           {metrics['bits_per_byte']:.4f}")
    
if __name__ == "__main__":
    main()
