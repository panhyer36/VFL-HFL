"""
LoRA (Low-Rank Adaptation) 模組 - 用於 HFL Transformer 的輕量級適配

將 LoRA 注入 HFL Transformer 內部層，替代外部 HFL Adapter，
實現更深層的 per-client 適配。

LoRA 目標層 (每個 TransformerEncoderLayer，共 4 層):
- Q proj: LoRA
- K proj: 凍結 (無 LoRA)
- V proj: LoRA
- out_proj: LoRA
- linear1 (FFN): LoRA
- linear2 (FFN): LoRA

初始化策略:
- lora_A: Kaiming 初始化
- lora_B: 零初始化 -> 初始時 LoRA 貢獻為 0，等同原始模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA 替代 nn.Linear。原始權重存為 buffer (不可訓練)，
    新增 lora_A / lora_B 為 Parameter。

    forward(x) = F.linear(x, weight, bias) + F.linear(F.linear(x, lora_A), lora_B) * scaling
    """

    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 原始權重存為 buffer (不可訓練)
        self.register_buffer('weight', original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None

        # LoRA 低秩矩陣
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Kaiming 初始化 lora_A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAMultiheadAttention(nn.Module):
    """
    LoRA 替代 nn.MultiheadAttention。
    將 packed in_proj_weight (3*d_model, d_model) 拆分為獨立 Q/K/V 投影:
    - Q -> LoRALinear (有 LoRA)
    - K -> buffer (凍結，無 LoRA)
    - V -> LoRALinear (有 LoRA)
    - out_proj -> LoRALinear (有 LoRA)

    手動實現 scaled dot-product attention，
    支持 batch_first=True 和 need_weights 參數。
    """

    def __init__(self, original_mha: nn.MultiheadAttention, rank: int, alpha: float):
        super().__init__()
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = original_mha.batch_first

        # 從 packed in_proj_weight 拆分 Q/K/V
        in_proj_weight = original_mha.in_proj_weight.data  # (3*d_model, d_model)
        in_proj_bias = original_mha.in_proj_bias.data if original_mha.in_proj_bias is not None else None
        d = self.embed_dim

        # Q projection -> LoRALinear
        q_linear = nn.Linear(d, d)
        q_linear.weight = nn.Parameter(in_proj_weight[:d].clone())
        if in_proj_bias is not None:
            q_linear.bias = nn.Parameter(in_proj_bias[:d].clone())
        self.q_proj = LoRALinear(q_linear, rank, alpha)

        # K projection -> buffer (凍結，無 LoRA)
        self.register_buffer('k_weight', in_proj_weight[d:2*d].clone())
        if in_proj_bias is not None:
            self.register_buffer('k_bias', in_proj_bias[d:2*d].clone())
        else:
            self.k_bias = None

        # V projection -> LoRALinear
        v_linear = nn.Linear(d, d)
        v_linear.weight = nn.Parameter(in_proj_weight[2*d:3*d].clone())
        if in_proj_bias is not None:
            v_linear.bias = nn.Parameter(in_proj_bias[2*d:3*d].clone())
        self.v_proj = LoRALinear(v_linear, rank, alpha)

        # out_proj -> LoRALinear
        out_linear = nn.Linear(d, d)
        out_linear.weight = nn.Parameter(original_mha.out_proj.weight.data.clone())
        if original_mha.out_proj.bias is not None:
            out_linear.bias = nn.Parameter(original_mha.out_proj.bias.data.clone())
        self.out_proj = LoRALinear(out_linear, rank, alpha)

        self.dropout = nn.Dropout(original_mha.dropout)

        # TransformerEncoderLayer.forward 會檢查這些屬性來判斷 fast path
        # 設為 None / False 確保走標準 _sa_block 路徑
        self.in_proj_bias = None
        self.in_proj_weight = None
        self._qkv_same_embed_dim = False

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):
        """
        與 nn.MultiheadAttention 相同的介面。
        TransformerEncoderLayer._sa_block() 呼叫:
            self_attn(x, x, x, attn_mask=..., key_padding_mask=...,
                      need_weights=False, is_causal=...)
        """
        # batch_first: (B, S, D) -> (S, B, D)
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        S, B, D = query.shape

        # Q, K, V 投影
        Q = self.q_proj(query)                                  # (S, B, D)
        K = F.linear(key, self.k_weight, self.k_bias)           # (S, B, D)
        V = self.v_proj(value)                                  # (S, B, D)

        # reshape for multi-head: (S, B, D) -> (B*num_heads, S, head_dim)
        Q = Q.contiguous().view(S, B * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.contiguous().view(S, B * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.contiguous().view(S, B * self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B*H, S, S)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, S) -> (B*H, 1, S)
            mask = key_padding_mask.unsqueeze(1).repeat(self.num_heads, 1, 1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, V)  # (B*H, S, head_dim)

        # reshape back: (B*H, S, head_dim) -> (S, B, D)
        attn_output = attn_output.transpose(0, 1).contiguous().view(S, B, D)

        # output projection
        attn_output = self.out_proj(attn_output)

        # batch_first: (S, B, D) -> (B, S, D)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            # 返回平均 head 的注意力權重: (B*H, S, S) -> (B, S, S)
            attn_weights = attn_weights.view(B, self.num_heads, S, S).mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


def apply_lora_to_transformer(model, rank, alpha):
    """
    遍歷 model.transformer.layers，替換:
    - self_attn -> LoRAMultiheadAttention
    - linear1 / linear2 -> LoRALinear
    並凍結所有非 LoRA 參數。

    Args:
        model: TransformerModel 實例
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha / rank)
    """
    if rank <= 0:
        return

    for layer in model.transformer.layers:
        # 替換 self_attn
        original_mha = layer.self_attn
        lora_mha = LoRAMultiheadAttention(original_mha, rank, alpha)
        layer.self_attn = lora_mha

        # 替換 linear1 (FFN 第一層)
        original_linear1 = layer.linear1
        layer.linear1 = LoRALinear(original_linear1, rank, alpha)

        # 替換 linear2 (FFN 第二層)
        original_linear2 = layer.linear2
        layer.linear2 = LoRALinear(original_linear2, rank, alpha)

    # 凍結所有非 LoRA 參數
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False


def get_lora_state_dict(model):
    """提取僅含 lora_A / lora_B 的 state_dict"""
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if 'lora_A' in name or 'lora_B' in name
    }


def load_lora_state_dict(model, state_dict):
    """載入 LoRA 權重"""
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"  Warning: LoRA key '{name}' not found in model")


def set_lora_trainable(model, trainable):
    """Phase 0 凍結 / Phase 1+ 解凍 LoRA 參數"""
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = trainable


def count_lora_params(model):
    """統計 LoRA 參數量"""
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return total, trainable
