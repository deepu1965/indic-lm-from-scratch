import argparse
import json
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from collections import defaultdict
from tokenizers import Tokenizer

from hindi.model.transformer import DecoderOnlyTransformer as HIDecoder
from hindi.model.config import ModelConfig as HIConfig
from nepali.model.transformer import DecoderOnlyTransformer as NEDecoder
from nepali.model.config import ModelConfig as NEConfig
from phase3.scripts.training.lora import apply_lora

def compute_metrics(attn_probs):
    """
    Computes mean entropy and mean distance for each head.
    attn_probs: shape (H, T, T) - post softmax probabilities
    """
    H_heads, T, _ = attn_probs.shape
    entropies = []
    distances = []
    for h in range(H_heads):
        head_attn = attn_probs[h]
        h_entropy = []
        h_distance = []
        for i in range(1, T):
            probs = head_attn[i, :i+1]
            ent = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
            h_entropy.append(ent)
            j_indices = torch.arange(i+1, device=probs.device, dtype=torch.float32)
            dist = torch.sum(probs * (i - j_indices)).item()
            h_distance.append(dist)
        entropies.append(np.mean(h_entropy))
        distances.append(np.mean(h_distance))
    return np.array(entropies), np.array(distances)

def get_hook(layer_idx, storage_dict):
    def hook(module, input, output):
        storage_dict[layer_idx] = output.detach().cpu()[0]
    return hook

def load_data(lang):
    jsonl_path = f"phase3/outputs/reasoning_{lang}.jsonl"
    test_records = []
    task_family_counts = defaultdict(int)
    viz_record = None
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("split") == "test":
                test_records.append(record)
                task_family_counts[record["task_family"]] += 1
                if record["id"] == "003010":
                    viz_record = record
                    
    assert len(test_records) == 700, f"Expected 700 test records, got {len(test_records)}"
    for family, count in task_family_counts.items():
        assert count == 100, f"Expected 100 records for task family {family}, got {count}"
    assert viz_record is not None, "Visualization record ID 003010 not found in test split"
    
    return test_records, viz_record

def load_model(lang, is_sft, device):
    base_model_cfg_path = f"{'hindi' if lang == 'hi' else 'nepali'}/configs/model_config.yaml"
    base_ckpt_path = f"{'hindi' if lang == 'hi' else 'nepali'}/outputs/checkpoints/best_model.pt"
    sft_ckpt_path = f"phase3/outputs/checkpoints/{lang}/best_model.pt"
    
    ConfigClass = HIConfig if lang == "hi" else NEConfig
    DecoderClass = HIDecoder if lang == "hi" else NEDecoder
    
    model_cfg = ConfigClass.from_yaml(base_model_cfg_path)
    model = DecoderClass(model_cfg).to(device)
    
    if is_sft:
        ckpt = torch.load(sft_ckpt_path, map_location=device)
        lora_cfg = ckpt["lora_config"]
        apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"])
        model.to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        ckpt = torch.load(base_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_cfg.num_layers, model_cfg.num_heads

def run_aggregate_analysis(model, test_records, tokenizer, num_layers, num_heads, device):
    attn_storage = {}
    hooks = []
    for l in [0, 11]:
        h = model.blocks[l].attention.attn_dropout.register_forward_hook(get_hook(l, attn_storage))
        hooks.append(h)
        
    agg_entropy = np.zeros((num_layers, num_heads))
    agg_distance = np.zeros((num_layers, num_heads))
    
    with torch.inference_mode():
        for record in test_records:
            input_tokens = tokenizer.encode(record["input_text"]).ids
            prompt = [2] + input_tokens + [3]
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            _ = model(input_ids)
            
            for l in [0, 11]:
                ent, dist = compute_metrics(attn_storage[l])
                agg_entropy[l] += ent
                agg_distance[l] += dist
                
    agg_entropy /= len(test_records)
    agg_distance /= len(test_records)
    
    for h in hooks:
        h.remove()
        
    return agg_entropy, agg_distance

def get_selected_heads(dists, ents):
    local_head = int(np.argmin(dists))
    long_head = int(np.argmax(dists))
    
    sorted_ent_heads = np.argsort(ents)
    focus_head = None
    for h in sorted_ent_heads:
        if h != local_head and h != long_head:
            focus_head = int(h)
            break
            
    return {
        "Local": local_head,
        "Long-range": long_head,
        "Focused": focus_head
    }

def generate_heatmap(attn_matrix, labels, title, out_file):
    mask = np.triu(np.ones_like(attn_matrix, dtype=bool), k=1)
    
    devanagari_font_path = "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"
    if not os.path.exists(devanagari_font_path):
        devanagari_font_path = "/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf"
    
    if os.path.exists(devanagari_font_path):
        fm.fontManager.addfont(devanagari_font_path)
        prop = fm.FontProperties(fname=devanagari_font_path, size=10)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        prop = fm.FontProperties(size=10)

    plt.figure(figsize=(18, 16))
    ax = sns.heatmap(attn_matrix, mask=mask, cmap="viridis", vmin=0.0, vmax=1.0,
                     xticklabels=labels, yticklabels=labels)
                     
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=prop, rotation=90, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop, rotation=0)
    
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("Key Tokens (Attended)", fontsize=14, labelpad=15)
    plt.ylabel("Query Tokens (Attending)", fontsize=14, labelpad=15)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=["hi", "ne"])
    args = parser.parse_args()
    
    lang = args.lang
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 3.2 Attention Analysis for {lang} on {device}")
    
    tokenizer_path = f"{'hindi' if lang == 'hi' else 'nepali'}/tokenizer/models/bpe_v10000.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    test_records, viz_record = load_data(lang)
    print("Data loaded successfully.")
    
    out_dir_base = f"phase3/outputs/evaluation/attention/{lang}"
    os.makedirs(f"{out_dir_base}/pretrained", exist_ok=True)
    os.makedirs(f"{out_dir_base}/sft", exist_ok=True)
    os.makedirs(f"{out_dir_base}/comparison/heatmaps", exist_ok=True)
    
    models_data = {}
    
    for condition in ["pretrained", "sft"]:
        print(f"\nEvaluating {condition} model...")
        model, num_layers, num_heads = load_model(lang, condition == "sft", device)
        
        agg_entropy, agg_distance = run_aggregate_analysis(model, test_records, tokenizer, num_layers, num_heads, device)
        
        metrics_list = []
        for l in [0, 11]:
            for h in range(num_heads):
                metrics_list.append({
                    "layer": l,
                    "head": h,
                    "mean_attention_entropy": float(agg_entropy[l, h]),
                    "mean_attention_distance": float(agg_distance[l, h])
                })
                
        with open(f"{out_dir_base}/{condition}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_list, f, indent=2)
            
        selected_heads = {}
        for l in [0, 11]:
            selected_heads[l] = get_selected_heads(agg_distance[l], agg_entropy[l])
            
        with open(f"{out_dir_base}/{condition}/selected_heads.json", "w", encoding="utf-8") as f:
            json.dump(selected_heads, f, indent=2)
            
        # Get heatmap matrix for viz_record
        attn_storage = {}
        hooks = []
        for l in [0, 11]:
            hooks.append(model.blocks[l].attention.attn_dropout.register_forward_hook(get_hook(l, attn_storage)))
            
        input_tokens = tokenizer.encode(viz_record["input_text"]).ids
        prompt = [2] + input_tokens + [3]
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        with torch.inference_mode():
            _ = model(input_ids)
            
        for h in hooks:
            h.remove()
            
        models_data[condition] = {
            "entropy": agg_entropy,
            "distance": agg_distance,
            "selected_heads": selected_heads,
            "viz_attn": attn_storage
        }
        
    print("\nSaving comparative artifacts...")
    comparison_results = {}
    for l in [0, 11]:
        comparison_results[l] = []
        for h in range(num_heads):
            pre_ent = models_data["pretrained"]["entropy"][l, h]
            sft_ent = models_data["sft"]["entropy"][l, h]
            pre_dist = models_data["pretrained"]["distance"][l, h]
            sft_dist = models_data["sft"]["distance"][l, h]
            comparison_results[l].append({
                "head": h,
                "pre_entropy": float(pre_ent),
                "sft_entropy": float(sft_ent),
                "delta_entropy": float(sft_ent - pre_ent),
                "pre_distance": float(pre_dist),
                "sft_distance": float(sft_dist),
                "delta_distance": float(sft_dist - pre_dist)
            })
            
    with open(f"{out_dir_base}/comparison/comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
        
    # Generate Heatmaps
    def get_token_label(t_id):
        if t_id == 4: return "<SPACE>"
        if t_id == 5: return "<NEWLINE>"
        s = tokenizer.id_to_token(t_id)
        return s if s is not None else f"[{t_id}]"
        
    labels = [get_token_label(t) for t in prompt]
    
    for l in [0, 11]:
        pre_sel = models_data["pretrained"]["selected_heads"][l]
        sft_sel = models_data["sft"]["selected_heads"][l]
        
        union_heads = set(pre_sel.values()).union(set(sft_sel.values()))
        
        for h in union_heads:
            roles = []
            for role, hid in pre_sel.items():
                if hid == h: roles.append(f"Pre_{role}")
            for role, hid in sft_sel.items():
                if hid == h: roles.append(f"SFT_{role}")
            roles_str = ",".join(roles)
            
            # Pretrained Heatmap
            title = f"{lang.upper()} Pretrained | L{l} H{h} ({roles_str})"
            out_file = f"{out_dir_base}/comparison/heatmaps/heatmap_l{l}_h{h}_pretrained.png"
            generate_heatmap(models_data["pretrained"]["viz_attn"][l].numpy()[h], labels, title, out_file)
            
            # SFT Heatmap
            title = f"{lang.upper()} SFT | L{l} H{h} ({roles_str})"
            out_file = f"{out_dir_base}/comparison/heatmaps/heatmap_l{l}_h{h}_sft.png"
            generate_heatmap(models_data["sft"]["viz_attn"][l].numpy()[h], labels, title, out_file)
            
    # Save qualitative metadata
    with open(f"{out_dir_base}/comparison/heatmaps/metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "language": lang,
            "example_id": viz_record["id"],
            "task_family": viz_record["task_family"],
            "difficulty": viz_record["difficulty"],
            "input_text": viz_record["input_text"],
            "token_ids": prompt,
            "token_labels": labels
        }, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
