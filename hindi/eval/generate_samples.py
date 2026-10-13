import argparse
import json
import torch
import time
import os
import random
from tokenizers import Tokenizer

from hindi.model import DecoderOnlyTransformer
from hindi.model.config import ModelConfig
from hindi.eval.generate_eval import generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", type=str, default="hindi/configs/model_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="hindi/outputs/checkpoints/best_model.pt")
    parser.add_argument("--tokenized-data", type=str, default="hindi/corpus/tokenised_splits/test.jsonl")
    parser.add_argument("--tokenizer-json", type=str, default="hindi/tokenizer/models/bpe_v10000.json")
    parser.add_argument("--out-json", type=str, default="hindi/eval/results/generated_samples.json")
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for hindi qualitative samples generation")

    # Reservoir sampling to deterministically select 20 examples
    random.seed(args.seed)
    reservoir = []
    count = 0
    
    print(f"Sampling {args.num_examples} documents from {args.tokenized_data}...")
    with open(args.tokenized_data, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            tokens = record.get("tokens", [])
            if len(tokens) >= args.prefix_len + args.max_new_tokens:
                if count < args.num_examples:
                    reservoir.append(record)
                else:
                    j = random.randint(0, count)
                    if j < args.num_examples:
                        reservoir[j] = record
                count += 1
                
    if not reservoir:
        raise ValueError("No valid documents found in the dataset.")
        
    print(f"Selected {len(reservoir)} documents out of {count} valid ones.")

    tokenizer = Tokenizer.from_file(args.tokenizer_json)

    model_cfg = ModelConfig.from_yaml(args.model_cfg)
    if args.prefix_len + args.max_new_tokens > model_cfg.context_length:
        raise ValueError("prefix_len + max_new_tokens > context_length")

    model = DecoderOnlyTransformer(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    # Set PyTorch seeds right before generation
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results = {
        "language": "hindi",
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_examples": len(reservoir),
        "prefix_len": args.prefix_len,
        "max_new_tokens": args.max_new_tokens,
        "examples": []
    }
    
    start_time = time.time()
    
    with torch.inference_mode():
        for idx, record in enumerate(reservoir):
            print(f"Processing example {idx+1}/{len(reservoir)}...")
            tokens = record.get("tokens", [])
        
            prefix_tokens = tokens[:args.prefix_len]
            ref_tokens = tokens[args.prefix_len:args.prefix_len + args.max_new_tokens]
        
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        
            example_data = {
                "example_id": idx,
                "document_id": record.get("id", "unknown"),
                "prefix": prefix_text,
                "reference_continuation": ref_text
            }
        
            input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        
            # Greedy
            gen_ids = generate(model, input_ids, args.max_new_tokens, temperature=0.0)
            gen_tokens = gen_ids[0, args.prefix_len:].tolist()
            example_data["greedy"] = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
            # Temp 0.5
            gen_ids = generate(model, input_ids, args.max_new_tokens, temperature=0.5)
            gen_tokens = gen_ids[0, args.prefix_len:].tolist()
            example_data["temperature_0.5"] = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
            # Temp 1.0
            gen_ids = generate(model, input_ids, args.max_new_tokens, temperature=1.0)
            gen_tokens = gen_ids[0, args.prefix_len:].tolist()
            example_data["temperature_1.0"] = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
            # Temp 1.5
            gen_ids = generate(model, input_ids, args.max_new_tokens, temperature=1.5)
            gen_tokens = gen_ids[0, args.prefix_len:].tolist()
            example_data["temperature_1.5"] = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
            results["examples"].append(example_data)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f}s.")
    print(f"Results saved to {args.out_json}")

if __name__ == "__main__":
    main()
