import json
import os
import torch
import argparse
from typing import Dict, Any

from hindi.model.transformer import DecoderOnlyTransformer as HIDecoder
from hindi.model.config import ModelConfig as HIConfig
from nepali.model.transformer import DecoderOnlyTransformer as NEDecoder
from nepali.model.config import ModelConfig as NEConfig
from phase3.scripts.training.lora import apply_lora
from tokenizers import Tokenizer

from phase3.scripts.evaluation.logical_resolver import resolve_expected_answer
from phase3.scripts.evaluation.exact_match_normalizer import normalize_for_match

@torch.inference_mode()
def generate_greedy(model, input_ids_tensor, max_new_tokens, eos_id):
    for _ in range(max_new_tokens):
        cond = input_ids_tensor[:, -model.config.context_length:]
        logits = model(cond)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids_tensor = torch.cat((input_ids_tensor, next_token), dim=-1)
        if next_token.item() == eos_id:
            break
    return input_ids_tensor

def extract_all_entities(generated_text: str, entity_mapping: dict, tokenizer: Tokenizer) -> list:
    found = []
    # Tokenizer decode() joins tokens with spaces.
    # By padding both generated text and entity names with spaces,
    # we use a conservative subword-boundary matching mechanism.
    padded_gen = f" {generated_text} "
    for eid, name in entity_mapping.items():
        name_decoded = tokenizer.decode(tokenizer.encode(name).ids)
        padded_name = f" {name_decoded} "
        if padded_name in padded_gen:
            found.append(eid)
    return found

def extract_predicted_entity(generated_text: str, entity_mapping: dict, tokenizer: Tokenizer):
    found = extract_all_entities(generated_text, entity_mapping, tokenizer)
    if len(found) == 1:
        return found[0]
    elif len(found) > 1:
        return "AMBIGUOUS"
    return None

def check_equality_semantics(generated_text: str, tokenizer: Tokenizer) -> bool:
    # Hindi and Nepali equality indicators
    eq_keywords = ["बराबर", "समान", "equal", "same"]
    padded_gen = f" {generated_text} "
    for w in eq_keywords:
        w_decoded = tokenizer.decode(tokenizer.encode(w).ids)
        padded_w = f" {w_decoded} "
        if padded_w in padded_gen:
            return True
    return False

def evaluate_record(record: dict, generated_text: str, tokenizer: Tokenizer) -> Dict[str, Any]:
    target_text = record["target_text"]
    
    # The tokenizer does not have a decoder, so decode(encode(text)) inserts spaces.
    # We must format the target_text and entities the exact same way to extract them.
    target_text_decoded = tokenizer.decode(tokenizer.encode(target_text).ids)
    
    is_exact_match = normalize_for_match(generated_text) == normalize_for_match(target_text_decoded)
    
    indep_resolved = resolve_expected_answer(record)
    expected_entity = indep_resolved["expected_answer_entity"]
    
    if expected_entity is None and indep_resolved.get("expected_relation") == "equal":
        predicted_entity = "EQUAL_CASE"
        expected_entities = set(record["question"]["entities"])
        found_entities = set(extract_all_entities(generated_text, record.get("entity_mapping", {}), tokenizer))
        
        has_eq = check_equality_semantics(generated_text, tokenizer)
        reasoning_correct = has_eq and (found_entities == expected_entities)
    else:
        predicted_entity = extract_predicted_entity(generated_text, record.get("entity_mapping", {}), tokenizer)
        if predicted_entity == "AMBIGUOUS" or predicted_entity is None:
            reasoning_correct = False
        else:
            reasoning_correct = (predicted_entity == expected_entity)
            
    return {
        "id": record["id"],
        "task_family": record["task_family"],
        "difficulty": record.get("difficulty", "unknown"),
        "input_text": record["input_text"],
        "target_text": target_text,
        "generated_text": generated_text,
        "independently_resolved_answer": expected_entity,
        "predicted_answer": predicted_entity,
        "reasoning_correct": reasoning_correct,
        "exact_match": is_exact_match
    }

def evaluate_model(lang, is_sft, max_examples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model_cfg_path = f"{'hindi' if lang == 'hi' else 'nepali'}/configs/model_config.yaml"
    base_ckpt_path = f"{'hindi' if lang == 'hi' else 'nepali'}/outputs/checkpoints/best_model.pt"
    tokenizer_path = f"{'hindi' if lang == 'hi' else 'nepali'}/tokenizer/models/bpe_v10000.json"
    sft_ckpt_path = f"phase3/outputs/checkpoints/{lang}/best_model.pt"
    
    ConfigClass = HIConfig if lang == "hi" else NEConfig
    DecoderClass = HIDecoder if lang == "hi" else NEDecoder
    
    model_cfg = ConfigClass.from_yaml(base_model_cfg_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    model = DecoderClass(model_cfg).to(device)
    
    ckpt_path = sft_ckpt_path if is_sft else base_ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if is_sft:
        base_ckpt = torch.load(base_ckpt_path, map_location=device)
        model.load_state_dict(base_ckpt['model_state_dict'], strict=True)
        lora_cfg = ckpt["lora_config"]
        apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"])
        model.to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    jsonl_path = f"phase3/outputs/reasoning_{lang}.jsonl"
    
    results = []
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("split") != "test": continue
            
            input_tokens = tokenizer.encode(record["input_text"]).ids
            prompt = [2] + input_tokens + [3]  # BOS ... EOS_sep
            prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
            
            out_tensor = generate_greedy(model, prompt_tensor, max_new_tokens=30, eos_id=3)
            
            gen_ids = out_tensor[0][len(prompt):].tolist()
            if 3 in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(3)]
            gen_text = tokenizer.decode(gen_ids)
            
            res = evaluate_record(record, gen_text, tokenizer)
            results.append(res)
            
            count += 1
            if max_examples and count >= max_examples:
                break
                
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=["hi", "ne"])
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    
    res = evaluate_model(args.lang, args.sft, args.max_examples)
    os.makedirs("phase3/outputs/evaluation", exist_ok=True)
    suffix = "_val" if args.max_examples is not None else ""
    out_path = f"phase3/outputs/evaluation/single_model_{args.lang}_{'sft' if args.sft else 'pre'}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Evaluated {len(res)} examples. Results saved to {out_path}")
