import argparse
import json
import torch
import time
import os
import sacrebleu
import string
from rouge_score import rouge_scorer
from tokenizers import Tokenizer

from hindi.model import DecoderOnlyTransformer
from hindi.model.config import ModelConfig

@torch.inference_mode()
def generate(model, input_ids, max_new_tokens, temperature=0.0):
    for _ in range(max_new_tokens):
        cond = input_ids[:, -model.config.context_length:]
        logits = model(cond)
        next_token_logits = logits[:, -1, :]
        if temperature == 0.0:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=-1)
    return input_ids

def get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

class IndicTokenizer:
    def tokenize(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation + '।॥'))
        return text.split()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenized-data", type=str, required=True)
    parser.add_argument("--tokenizer-json", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {device} for hindi generation")

    tokenizer = Tokenizer.from_file(args.tokenizer_json)

    model_cfg = ModelConfig.from_yaml(args.model_cfg)
    if args.prefix_len + args.max_new_tokens > model_cfg.context_length:
        raise ValueError("prefix_len + max_new_tokens > context_length")

    model = DecoderOnlyTransformer(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    hyps = []
    refs = []
    
    docs_processed = 0
    start_time = time.time()
    
    print(f"Streaming data from {args.tokenized_data}...")
    with open(args.tokenized_data, "r", encoding="utf-8") as f:
        for line in f:
            if docs_processed >= args.num_examples:
                break
                
            record = json.loads(line)
            tokens = record.get("tokens", [])
            
            if len(tokens) < args.prefix_len + args.max_new_tokens:
                continue
                
            prefix = tokens[:args.prefix_len]
            ref = tokens[args.prefix_len:args.prefix_len + args.max_new_tokens]
            
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            generated_ids = generate(model, input_ids, args.max_new_tokens, temperature=args.temperature)
            
            gen_tokens = generated_ids[0, args.prefix_len:].tolist()
            
            # skip_special_tokens=True ensures no BOS/EOS contamination
            decoded_gen = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded_ref = tokenizer.decode(ref, skip_special_tokens=True)
            
            hyps.append(decoded_gen)
            refs.append(decoded_ref)
            
            docs_processed += 1
            if docs_processed % 50 == 0:
                print(f"Generated {docs_processed}/{args.num_examples} documents...")

    if docs_processed == 0:
        raise ValueError("No valid documents found!")

    print("Computing metrics...")
    
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=IndicTokenizer())
    rouge_scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    total_1grams = 0
    unique_1grams = set()
    total_2grams = 0
    unique_2grams = set()
    total_4grams = 0
    unique_4grams = set()

    for hyp in hyps:
        words = hyp.split()
        
        g1 = get_ngrams(words, 1)
        total_1grams += len(g1)
        unique_1grams.update(g1)
        
        g2 = get_ngrams(words, 2)
        total_2grams += len(g2)
        unique_2grams.update(g2)
        
        g4 = get_ngrams(words, 4)
        total_4grams += len(g4)
        unique_4grams.update(g4)
        
    distinct_1 = len(unique_1grams) / total_1grams if total_1grams > 0 else 0.0
    distinct_2 = len(unique_2grams) / total_2grams if total_2grams > 0 else 0.0
    repetition_rate = 1.0 - (len(unique_4grams) / total_4grams) if total_4grams > 0 else 0.0

    metrics = {
        "language": "hindi",
        "split": args.split,
        "generation_method": "greedy" if args.temperature == 0.0 else "temperature",
        "temperature": args.temperature if args.temperature > 0.0 else "N/A",
        "seed": args.seed,
        "prefix_len": args.prefix_len,
        "max_new_tokens": args.max_new_tokens,
        "num_examples": docs_processed,
        "bleu": bleu,
        "chrf": chrf,
        "rouge_l": rouge_l,
        "repetition_rate": repetition_rate,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    csv_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", encoding="utf-8") as f:
        if not csv_exists:
            f.write("language,split,generation_method,temperature,seed,prefix_len,max_new_tokens,num_examples,bleu,chrf,rouge_l,repetition_rate,distinct_1,distinct_2\n")
        f.write(f"{metrics['language']},{metrics['split']},{metrics['generation_method']},{metrics['temperature']},{metrics['seed']},{metrics['prefix_len']},{metrics['max_new_tokens']},{metrics['num_examples']},{metrics['bleu']:.4f},{metrics['chrf']:.4f},{metrics['rouge_l']:.4f},{metrics['repetition_rate']:.4f},{metrics['distinct_1']:.4f},{metrics['distinct_2']:.4f}\n")
        
    elapsed = time.time() - start_time
    print(f"Evaluated {docs_processed} examples in {elapsed:.2f}s")
    print(f"BLEU-4: {bleu:.2f} | chrF: {chrf:.2f} | ROUGE-L: {rouge_l:.4f}")
    print(f"RepRate: {repetition_rate:.4f} | Dist-1: {distinct_1:.4f} | Dist-2: {distinct_2:.4f}")
    
if __name__ == "__main__":
    main()
