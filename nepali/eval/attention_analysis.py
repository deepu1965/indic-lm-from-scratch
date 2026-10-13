import argparse
import json
import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
import matplotlib.font_manager as fm

def compute_metrics(attn_probs):
    """
    Computes mean entropy and mean distance for each head.
    attn_probs: shape (H, T, T) - post softmax probabilities
    """
    H_heads, T, _ = attn_probs.shape
    
    entropies = []
    distances = []
    
    for h in range(H_heads):
        head_attn = attn_probs[h] # (T, T)
        
        h_entropy = []
        h_distance = []
        
        # Skip token 0 (trivial attention to itself)
        for i in range(1, T):
            probs = head_attn[i, :i+1] # valid probabilities up to i
            
            # Entropy: -sum(p * log2(p + eps))
            ent = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
            h_entropy.append(ent)
            
            # Distance: sum(p * (i - j))
            j_indices = torch.arange(i+1, device=probs.device, dtype=torch.float32)
            dist = torch.sum(probs * (i - j_indices)).item()
            h_distance.append(dist)
            
        entropies.append(np.mean(h_entropy))
        distances.append(np.mean(h_distance))
        
    return np.array(entropies), np.array(distances)

def get_hook(layer_idx, storage_dict):
    def hook(module, input, output):
        # output is the post-dropout probabilities: (B, H, T, T)
        # B=1, so we take index 0
        storage_dict[layer_idx] = output.detach().cpu()[0]
    return hook

def stream_100_sequences(filepath, seq_len=128, max_seqs=100):
    seqs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            tokens = rec.get("tokens", [])
            if len(tokens) >= seq_len:
                seqs.append(tokens[:seq_len])
                if len(seqs) >= max_seqs:
                    break
    return seqs

def stream_find(filepath, target_id):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") == target_id:
                return rec
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["hindi", "nepali"], required=True)
    args = parser.parse_args()
    
    lang = args.language
    
    model_cfg_path = f"{lang}/configs/model_config.yaml"
    checkpoint_path = f"{lang}/outputs/checkpoints/best_model.pt"
    tokenizer_path = f"{lang}/tokenizer/models/bpe_v10000.json"
    tok_data_path = f"{lang}/corpus/tokenised_splits/test.jsonl"
    out_dir = f"{lang}/eval/results/heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    
    if lang == "hindi":
        viz_doc_id = "id_711f0cc13922969a1335f1299cea273a117af82c07a105a7909c53a21a857ef1"
    else:
        viz_doc_id = "id_a9b395f0f4b13ffce03324d1894041051a202f25f943e5956a527b22738ccc60"
        
    sys.path.insert(0, os.getcwd())
    from hindi.model import DecoderOnlyTransformer
    from hindi.model.config import ModelConfig
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {lang} model on {device}...")
    
    cfg = ModelConfig.from_yaml(model_cfg_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    model = DecoderOnlyTransformer(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    
    print(f"Streaming 100 sequences of 128 tokens from {tok_data_path}...")
    seqs = stream_100_sequences(tok_data_path, seq_len=128, max_seqs=100)
    if len(seqs) < 100:
        print(f"Warning: Only found {len(seqs)} valid sequences.")
        
    num_layers = cfg.num_layers
    num_heads = cfg.num_heads
    
    attn_storage = {}
    hooks = []
    for l in range(num_layers):
        h = model.blocks[l].attention.attn_dropout.register_forward_hook(get_hook(l, attn_storage))
        hooks.append(h)
        
    agg_entropy = np.zeros((num_layers, num_heads))
    agg_distance = np.zeros((num_layers, num_heads))
    
    print("Computing aggregate metrics over all layers and heads...")
    with torch.inference_mode():
        for idx, tokens in enumerate(seqs):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            _ = model(input_ids)
            
            for l in range(num_layers):
                ent, dist = compute_metrics(attn_storage[l])
                agg_entropy[l] += ent
                agg_distance[l] += dist
                
    agg_entropy /= len(seqs)
    agg_distance /= len(seqs)
    
    metrics_csv_path = f"{lang}/eval/results/attention_metrics.csv"
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        f.write("layer,head,mean_attention_entropy,mean_attention_distance\n")
        for l in range(num_layers):
            for h in range(num_heads):
                f.write(f"{l},{h},{agg_entropy[l, h]:.4f},{agg_distance[l, h]:.4f}\n")
    print(f"Saved aggregate metrics to {metrics_csv_path}")
    
    selected_heads = {0: {}, 11: {}}
    selected_heads_path = f"{lang}/eval/results/selected_attention_heads.txt"
    
    with open(selected_heads_path, "w", encoding="utf-8") as f_sel:
        for layer_idx in [0, 11]:
            dists = agg_distance[layer_idx]
            ents = agg_entropy[layer_idx]
            
            local_head = int(np.argmin(dists))
            long_head = int(np.argmax(dists))
            
            sorted_ent_heads = np.argsort(ents)
            focus_head = None
            for h in sorted_ent_heads:
                if h != local_head and h != long_head:
                    focus_head = int(h)
                    break
                    
            selected_heads[layer_idx] = {
                "Local": local_head,
                "Long-range": long_head,
                "Focused": focus_head
            }
            
            f_sel.write(f"Layer {layer_idx}\n")
            f_sel.write("-" * len(f"Layer {layer_idx}") + "\n")
            f_sel.write("Local Head:\n")
            f_sel.write(f"Head {local_head}\n")
            f_sel.write(f"Mean Attention Distance: {dists[local_head]:.4f}\n")
            f_sel.write(f"Mean Attention Entropy: {ents[local_head]:.4f}\n\n")
            f_sel.write("Long-range Head:\n")
            f_sel.write(f"Head {long_head}\n")
            f_sel.write(f"Mean Attention Distance: {dists[long_head]:.4f}\n")
            f_sel.write(f"Mean Attention Entropy: {ents[long_head]:.4f}\n\n")
            f_sel.write("Focused Head:\n")
            f_sel.write(f"Head {focus_head}\n")
            f_sel.write(f"Mean Attention Distance: {dists[focus_head]:.4f}\n")
            f_sel.write(f"Mean Attention Entropy: {ents[focus_head]:.4f}\n\n\n")
            
            print(f"\nLayer {layer_idx}:")
            print(f"Head {local_head} -> Local: distance={dists[local_head]:.2f}, entropy={ents[local_head]:.2f}")
            print(f"Head {long_head} -> Long-range: distance={dists[long_head]:.2f}, entropy={ents[long_head]:.2f}")
            print(f"Head {focus_head} -> Focused: distance={dists[focus_head]:.2f}, entropy={ents[focus_head]:.2f}")
            
    print(f"Saved selected heads to {selected_heads_path}")
        
    for h in hooks:
        h.remove()
    attn_storage.clear()
    
    model.blocks[0].attention.attn_dropout.register_forward_hook(get_hook(0, attn_storage))
    model.blocks[11].attention.attn_dropout.register_forward_hook(get_hook(11, attn_storage))
    
    print(f"\nExtracting visualization document {viz_doc_id}...")
    viz_rec = stream_find(tok_data_path, viz_doc_id)
    if not viz_rec:
        raise ValueError(f"Could not find visualization document for {lang}!")
        
    viz_tokens = viz_rec["tokens"][:48]
    
    def get_token_label(t_id):
        if t_id == 4: return "<SPACE>"
        if t_id == 5: return "<NEWLINE>"
        s = tokenizer.id_to_token(t_id)
        return s if s is not None else f"[{t_id}]"
        
    labels = [get_token_label(t) for t in viz_tokens]
    
    with torch.inference_mode():
        input_ids = torch.tensor([viz_tokens], dtype=torch.long, device=device)
        _ = model(input_ids)
        
    devanagari_font_path = "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"
    if not os.path.exists(devanagari_font_path):
        devanagari_font_path = "/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf"
    
    if os.path.exists(devanagari_font_path):
        fm.fontManager.addfont(devanagari_font_path)
        prop = fm.FontProperties(fname=devanagari_font_path, size=10)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        prop = fm.FontProperties(size=10)
            
    print("Generating heatmaps...")
    for layer_idx in [0, 11]:
        layer_attn = attn_storage[layer_idx].numpy() # (H, 48, 48)
        
        for role, head_idx in selected_heads[layer_idx].items():
            matrix = layer_attn[head_idx] # (48, 48)
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            
            plt.figure(figsize=(18, 16))
            ax = sns.heatmap(matrix, mask=mask, cmap="viridis", vmin=0.0, vmax=1.0, 
                             xticklabels=labels, yticklabels=labels)
                             
            ax.set_xticklabels(ax.get_xticklabels(), fontproperties=prop, rotation=90, ha='right', rotation_mode='anchor')
            ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop, rotation=0)
            
            plt.title(f"{lang.capitalize()} | Layer {layer_idx} | Head {head_idx} ({role})", fontsize=18, pad=20)
            plt.xlabel("Key Tokens (Attended)", fontsize=14, labelpad=15)
            plt.ylabel("Query Tokens (Attending)", fontsize=14, labelpad=15)
            plt.tight_layout()
            
            out_file = os.path.join(out_dir, f"heatmap_layer{layer_idx}_head{head_idx}_{role.lower()}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"Saved {out_file}")

    print("\nAttention analysis and heatmap generation fully completed.")

if __name__ == "__main__":
    main()
