from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_var import FFN, SelfAttention, AdaLNSelfAttn, AdaLNBeforeHead


def attention_forward(self, x, attn_bias):
    B, L, C = x.shape
    
    qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
    
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
    oup = self.matmul2(attn.softmax(dim=-1), v)
    oup = oup.transpose(1, 2).reshape(B, L, C)
    
    return self.proj_drop(self.proj(oup))


class MatMul(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, A, B):
        return A @ B


def get_net(model):
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            setattr(module, "matmul1", MatMul("matmul1"))
            setattr(module, "matmul2", MatMul("matmul2"))
            module.forward = MethodType(attention_forward, module)

    model.cuda()
    model.eval()
    return model