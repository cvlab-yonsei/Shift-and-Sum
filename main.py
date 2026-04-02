import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VQVAE, build_vae_var
import PIL.Image as PImage
from tqdm import tqdm
import dist

from quant import *
from optim import *

def run(args):
    # Build full-precision model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{args.model_depth}.pth'
    vae_ckpt, var_ckpt = os.path.join(args.weight_dir, vae_ckpt), os.path.join(args.weight_dir, var_ckpt)
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )

    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # Quantize model
    np.random.seed(args.seed)
    calib_labels = np.random.choice(1000, args.calib_size, replace=False)
    calib_labels = torch.from_numpy(calib_labels).type(torch.LongTensor)
    calib_labels = calib_labels.split(1)

    q_var = init_quant_model(var, calib_labels, args)
    print(f'init complete.')

    kwargs = dict(labels = calib_labels, batch_size = args.batch_size, iters = args.iters, weight=args.weight, opt_mode = 'mse', p = args.p, lr = args.lr, wwq=args.wwq, waq=args.waq, cfg=args.cfg, seed = args.seed)

    # Shift-and-sum quantization
    total_bitops = compute_total_bitop(q_var, args)
    thres = get_thres_with_bitop(q_var, calib_labels, total_bitops, args)
    for module in q_var.modules():
        if isinstance(module, QuantMatMul):
            module.theta = thres

    # Calibration data resampling
    from models.helpers import convert_logit_to_prob
    rng = q_var.model.rng
    V = len(var.vae_quant_proxy[0].embedding.weight)
    get_logit = GetLayerLogit(q_var, device, args.cfg, args.seed)

    idx_Bls = torch.empty(args.calib_size, 0).long().to(device)
    tmp_idx_Bls = [torch.empty(1, 0).long().to(device) for _ in range(args.calib_size)]
    prob_lVs = [torch.empty(0, V).to(device) for _ in range(args.calib_size)]

    bal_rngs = []
    for label_B in calib_labels:
        seed = args.seed + 10000 + label_B.item()
        g = torch.Generator(device=dist.get_device())
        g.manual_seed(seed)
        bal_rngs.append(g)

    for i, label_B in enumerate(calib_labels):
        rng.manual_seed(args.seed + label_B.item())
        rng_state = rng.get_state()
        for pn in patch_nums:
            logits_BlV = get_logit(label_B, tmp_idx_Bls[i])
            prob_lV = convert_logit_to_prob(logits_BlV, top_k=900, top_p=0.96)
            assigned_idx_Bl = torch.multinomial(prob_lV, num_samples=1, replacement=True, generator=rng).view(1, pn ** 2)
            prob_lVs[i] = torch.cat([prob_lVs[i], prob_lV], dim=0)
            tmp_idx_Bls[i] = torch.cat([tmp_idx_Bls[i], assigned_idx_Bl], dim=1)
        rng.set_state(rng_state)

    prob_BlV = torch.stack(prob_lVs, dim=0)
    tmp_idx_Bls = torch.cat(tmp_idx_Bls, dim=0)

    cur_L = 0
    for pn in patch_nums:
        cur_L += pn*pn
        prob = prob_BlV[:, cur_L-pn*pn:cur_L]
        sampled = tmp_idx_Bls[:, cur_L-pn*pn:cur_L].clone()

        prob_mean = prob.mean(dim=(0, 1))
        N = sampled.numel()
        target_count = (prob_mean * N).floor().long()
        remainder = N - target_count.sum().item()

        if remainder > 0:
            fractional = (prob_mean * N) - target_count.float()
            topk_indices = torch.topk(fractional, k=remainder).indices
            target_count[topk_indices] += 1

        current_count = torch.bincount(sampled.flatten(), minlength=V)
        excess = (current_count - target_count).clamp(min=0)
        shortage = (target_count - current_count).clamp(min=0)

        over_classes = (excess > 0).nonzero(as_tuple=True)[0].tolist()
        under_classes = (shortage > 0).nonzero(as_tuple=True)[0].tolist()

        for c in over_classes:
            token_idx = (sampled == c).nonzero(as_tuple=False)
            if len(token_idx) == 0:
                continue
            b_idx, l_idx = token_idx[:,0], token_idx[:,1]
            n_excess = excess[c].item()
            if n_excess == 0:
                continue
            perm = prob[b_idx, l_idx][:, under_classes].sum(dim=-1)
            perm = torch.argsort(perm, dim=0, descending=True)[:n_excess]
            reassign_b, reassign_l = token_idx[perm][:,0], token_idx[perm][:,1]

            for b, l in zip(reassign_b, reassign_l):
                prob_i = prob[b, l]
                mask = torch.zeros_like(prob_i)
                mask[under_classes] = 1
                masked_prob = prob_i * mask
                if masked_prob.sum().item() == 0:
                    masked_prob = torch.ones_like(prob_i) * mask
                bal_rng = bal_rngs[b]
                new_cls = torch.multinomial(masked_prob, num_samples=1, generator=bal_rng).item()

                sampled[b, l] = new_cls
                shortage[new_cls] -= 1
                if shortage[new_cls] == 0:
                    under_classes.remove(new_cls)
        idx_Bls = torch.cat([idx_Bls, sampled], dim=1)
    kwargs['idx_Bls'] = idx_Bls.split(1)

    def recon_model(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, block_for_recon):
                print('Reconstruction for block {}'.format(name))
                block_reconstruction(q_var, module, **kwargs)
            elif isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                layer_reconstruction(q_var, module, **kwargs)
            else:
                recon_model(module)

    # Recon model
    recon_model(q_var)

    # Shift-and-sum quantization
    thres = get_thres_with_bitop(q_var, calib_labels, total_bitops, args)
    for module in q_var.modules():
        if isinstance(module, QuantMatMul):
            module.theta = thres

    # Sample
    q_var.set_quant_state(True, True)
    img_per_class = args.n_samples // 1000
    img_per_iter = 5 # can be changed according to hardware constraints
    assert img_per_class % img_per_iter == 0

    split_list = torch.tensor([img_per_iter] * (img_per_class // img_per_iter))
    split_list = split_list.repeat(1000)
    labels = torch.arange(1000).repeat(img_per_class).type(torch.LongTensor)
    labels = labels.split(split_list.tolist())
    cnt = 0
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            for i, label_B in enumerate(tqdm(labels, desc='sample')):
                recon_B3HW = q_var.autoregressive_infer_cfg(B=label_B.size(0), label_B=label_B.to(device), cfg=args.cfg, top_k=900, top_p=0.96, g_seed=args.seed + i, more_smooth=False)
                for b in range(label_B.size(0)):
                    cnt += 1
                    chw = recon_B3HW[b].clamp(min=0.0, max=1.0).permute(1, 2, 0).mul_(255).cpu().numpy()
                    chw = PImage.fromarray(chw.astype(np.uint8)).save(os.path.join(args.save_dir, f"{cnt:05}.png"))


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Official Implementation of Shift-and-Sum Quantization for Visual Autoregressive Models')
    parser.add_argument("--weight_dir", type=str, default='/path/to/weight', help='weight directory')
    parser.add_argument("--save_dir", type=str, default='/path/to/gen_imgs', help='place to store the generated images')
    parser.add_argument("--model_depth", type=int, default=16, help='model depth')
    parser.add_argument("--seed", type=int, default=2025, help='seed to use')
    parser.add_argument("--device", type=str, default='0', help='device to use')
    parser.add_argument("--n_samples", type=int, default=50000, help='number of samples to be generated')
    parser.add_argument("--cfg", type=int, default=1.5, help='classifier-free guidance')
    # quantization configs
    parser.add_argument("--w_bit", type=int, default=8, help='bit setting for weights')
    parser.add_argument("--a_bit", type=int, default=8, help='bit setting for activations')
    parser.add_argument("--channel_wise", action='store_true', help='use channel-wise weight quantization')
    parser.add_argument("--head_wise", action='store_true', help='use head-wise quantization for matmul')
    parser.add_argument("--scale_method", type=str, default='percentile', help='scaling method')
    parser.add_argument("--eps", type=float, default=1e-5, help='for initialization of activation quant params')
    # optimization configs
    parser.add_argument("--calib_size", type=int, default=256, help='calib size')
    parser.add_argument("--batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--iters", type=int, default=5000, help='number of iterations')
    parser.add_argument("--lr", type=float, default=4e-4, help='learning rate')
    parser.add_argument("--p", type=int, default=2, help='L_p norm minimization')
    parser.add_argument("--wwq", action='store_true', help='use weight quantization during adaround')
    parser.add_argument("--waq", action='store_true', help='use activation quantization during adaround')
    parser.add_argument("--weight", type=float, default=0.001, help='adaround weight')
    # shift-and-sum quantization
    parser.add_argument("--alpha", type=float, default=0.01, help='bitop constraint')
    parser.add_argument("--n_grids", type=int, default=10000, help='number of grids for searching')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.device
    run(args)