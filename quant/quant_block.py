import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath
from quant.quant_layer import QuantModule, QuantMatMul
from models.basic_var import FFN, SelfAttention, AdaLNSelfAttn, AdaLNBeforeHead

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func
except ImportError: pass


def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: attn.add_(attn_mask)
    return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class BaseQuantBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, QuantMatMul):
                m.set_quant_state(act_quant, act_quant)


class QFFN(BaseQuantBlock):
    def __init__(self, module: FFN, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}):
        super().__init__()
        self.fc1 = QuantModule(module.fc1, weight_quant_params, act_quant_params)
        self.act = module.act
        self.fc2 = QuantModule(module.fc2, weight_quant_params, act_quant_params)
        self.drop = module.drop
        self.fc2.act_quantizer.fc2 = True

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class QSelfAttention(BaseQuantBlock):
    def __init__(self, module: SelfAttention, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}):
        super().__init__()
        self.block_idx, self.num_heads, self.head_dim = module.block_idx, module.num_heads, module.head_dim
        self.attn_l2_norm = module.attn_l2_norm
        self.scale = module.scale
        if self.attn_l2_norm:
            self.scale_mul_1H11 = module.scale_mul_1H11
            self.max_scale_mul = module.max_scale_mul

        self.mat_qkv = QuantModule(module.mat_qkv, weight_quant_params, act_quant_params)
        self.q_bias, self.v_bias = module.q_bias, module.v_bias
        self.zero_k_bias = module.zero_k_bias
        self.mat_qkv.bias = self.mat_qkv.org_bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))

        self.matmul1 = QuantMatMul('matmul1', matmul_params)
        self.matmul2 = QuantMatMul('matmul2', matmul_params)

        self.proj = QuantModule(module.proj, weight_quant_params, act_quant_params)
        self.proj_drop = module.proj_drop
        self.attn_drop = module.attn_drop

        self.using_flash = module.using_flash
        self.using_xform = module.using_xform

        # only used during inference
        self.caching, self.cached_k, self.cached_v = module.caching, module.cached_k, module.cached_v

    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, attn_bias=None):
        B, L, C = x.shape

        qkv = self.mat_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        attn = self.matmul1(q.mul(self.scale), k.transpose(-2, -1)) # BHLc @ BHcL => BHLL
        if attn_bias is not None: attn.add_(attn_bias)

        oup = self.matmul2(attn.softmax(dim=-1), v) # BHL1L2 @ BHL2c
        oup = oup.transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
        

class QAdaLNSelfAttn(BaseQuantBlock):
    def __init__(self, module: AdaLNSelfAttn, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}):
        super().__init__()
        self.block_idx, self.last_drop_p = module.block_idx, module.last_drop_p
        self.C, self.D = module.C, module.D
        self.drop_path = module.drop_path
        
        self.ln_wo_grad = module.ln_wo_grad
        self.shared_aln = module.shared_aln
        if self.shared_aln:
            self.ada_gss = module.ada_gss
        else:
            lin = QuantModule(module.ada_lin[1], weight_quant_params, act_quant_params)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.attn = QSelfAttention(module.attn, weight_quant_params, act_quant_params, matmul_params)
        self.ffn = QFFN(module.ffn, weight_quant_params, act_quant_params, matmul_params)
        
        self.fused_add_norm_fn = None
        self.recon_err = []
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias=None):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x


class QAdaLNBeforeHead(BaseQuantBlock):
    def __init__(self, module: AdaLNBeforeHead, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}):
        super().__init__()
        self.C, self.D = module.C, module.D
        self.ln_wo_grad = module.ln_wo_grad
        lin = QuantModule(module.ada_lin[1], weight_quant_params, act_quant_params)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


block_specials = {
    AdaLNSelfAttn: QAdaLNSelfAttn,
}

block_for_recon = (QFFN, QSelfAttention)