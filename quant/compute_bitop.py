import torch
import torch.nn.functional as F
from quant.quant_layer import QuantModule, QuantMatMul


class AttnHook:
    def __init__(self):
        self.attn_length = []
        self.attn_store = []

    def __call__(self, module, A, B):
        self.attn_length.append(A[0].size(2))
        attn_store = A[0].mean(dim=2)
        self.attn_store.append(attn_store)

    def clear(self):
        self.attn_store = []


@torch.no_grad()
def get_thres_with_bitop(q_var, calib_labels, total_bitops, args):
    attn_hooks, handles = [], []
    for block in q_var.model.blocks:
        attn_hook = AttnHook()
        handle = block.attn.matmul2.register_forward_hook(attn_hook)
        attn_hooks.append(attn_hook)
        handles.append(handle)

    q_var.set_quant_state(True, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_var.get_rng_state()
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            _ = q_var.autoregressive_infer_cfg(B=calib_labels[0].size(0), label_B=calib_labels[0].to(device), cfg=args.cfg, top_k=900, top_p=0.96, g_seed=args.seed, more_smooth=False)
    q_var.set_rng_state()

    head_dim = q_var.model.blocks[0].attn.head_dim

    bitops_list = []
    max_log_tokens = 3
    for theta in torch.linspace(1 / args.n_grids, 1, args.n_grids):
        bitops = 0

        for attn_hook in attn_hooks:
            for s1, attn in zip(attn_hook.attn_length, attn_hook.attn_store):
                b, h, s2 = attn.size()
                log_n_tokens = torch.log2(attn / theta).ceil().clamp(min=0, max=max_log_tokens)
                n_tokens = torch.sum(2 ** log_n_tokens - 1)

                mean_bitops = b * h * s1 * s2 * 16
                count_bitops = b * h * s2 * 16 + b * h * s2 * (max_log_tokens + 1)
                add_bitops = n_tokens * head_dim * 16 + n_tokens * s1
                matmul_bitops = s1 * n_tokens * head_dim * (args.a_bit ** 2)
                bitops += (mean_bitops + count_bitops + matmul_bitops + add_bitops) / b
                
        bitops_list.append(bitops)
    thres = (torch.tensor(bitops_list) > args.alpha * total_bitops).sum().item()
    assert bitops_list[0] > args.alpha * total_bitops, "Insufficient n_grid or too high BOP constraint"

    for handle in handles:
        handle.remove()

    return torch.linspace(1 / args.n_grids, 1, args.n_grids)[thres]


@torch.no_grad()
def compute_total_bitop(q_var, args):
    w_bit, a_bit = args.w_bit, args.a_bit
    num_heads, head_dim = q_var.model.blocks[0].attn.num_heads, q_var.model.blocks[0].attn.head_dim
    num_stages = len(q_var.patch_nums)
    total_bitops = 0

    # Compute bitops for each transformer block
    for name, module in q_var.named_modules():
        if isinstance(module, QuantModule):
            if module.fwd_func == F.linear:
                fn, c = module.weight.size()
                multiplier = num_stages if name.split('.')[-2] == 'ada_lin' else 1
                bitops = 0
                for patch_num in module.patch_nums:
                    bitops += fn * c * patch_num * w_bit * a_bit * multiplier
                total_bitops += bitops

        elif isinstance(module, QuantMatMul):
            if module.mode == 'matmul1':
                bitops = 0
                for A_patch_num, B_patch_num in zip(module.A_patch_nums, module.B_patch_nums):
                    bitops += num_heads * A_patch_num * B_patch_num * (a_bit ** 2) * head_dim
                total_bitops += bitops

            elif module.mode == 'matmul2':
                bitops = 0
                for A_patch_num, B_patch_num in zip(module.A_patch_nums, module.B_patch_nums):
                    bitops += num_heads * A_patch_num * B_patch_num * (a_bit ** 2) * head_dim
                total_bitops += bitops

    for module in q_var.model.vae_quant_proxy[0].quant_resi.qresi_ls:
        if isinstance(module, QuantModule):
            if module.fwd_func == F.conv2d:
                fn, c, fh, fw = module.weight.size()
                s, p, d, g = module.fwd_kwargs['stride'], module.fwd_kwargs['padding'], module.fwd_kwargs['dilation'], module.fwd_kwargs['groups']
                bitops = 0
                for patch_num in module.patch_nums:
                    bitops += fn * (c // g) * (patch_num // (s[0] * s[1])) * fh * fw * w_bit * a_bit
                total_bitops += bitops
    return total_bitops
