import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds LoRA A and B matrices.
    
    The base linear layer is frozen.
    The output is: Wx + (B A x) * scaling
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        
        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        # Scaling
        self.scaling = alpha / rank
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming uniform for A (matches default nn.Linear init for inputs)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Zero for B so that initial output is identical to base_layer
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        base_out = self.base_layer(x)
        
        # LoRA forward pass: x -> dropout -> A -> B -> scaling
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        
        return base_out + lora_out * self.scaling


def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
    """
    Walks the model and replaces target attention projections with LoRALinear.
    Freezes all non-LoRA parameters.
    
    Target modules: blocks.{i}.attention.{W_q, W_k, W_v, W_o}
    """
    # Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False
        
    # We must support both Hindi and Nepali model imports for CausalSelfAttention
    # Since they share the same architecture definition in python, checking class name is robust.
    for name, module in model.named_modules():
        if module.__class__.__name__ == "CausalSelfAttention":
            # Replace W_q
            module.W_q = LoRALinear(module.W_q, rank, alpha, dropout)
            # Replace W_k
            module.W_k = LoRALinear(module.W_k, rank, alpha, dropout)
            # Replace W_v
            module.W_v = LoRALinear(module.W_v, rank, alpha, dropout)
            # Replace W_o
            module.W_o = LoRALinear(module.W_o, rank, alpha, dropout)
            
    # Verify trainable parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters (base + LoRA): {total:,}")
    print(f"Trainable LoRA parameters: {trainable:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")


def create_lora_optimizer(model: nn.Module, learning_rate: float, weight_decay: float = 0.01) -> AdamW:
    """
    Creates an AdamW optimizer containing ONLY the trainable LoRA parameters.
    Since LoRA parameters are exclusively weight matrices (no biases/LayerNorms),
    we apply weight decay to all of them uniformly.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optim_groups = [
        {"params": trainable_params, "weight_decay": weight_decay}
    ]
    
    return AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
