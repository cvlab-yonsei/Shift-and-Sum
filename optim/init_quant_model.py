import torch
import numpy as np

from quant.quantizer import UniformAffineQuantizer, GroupAffineQuantizer
from quant.quant_layer import QuantModule, QuantMatMul
from quant.quant_model import QuantVAR
from quant.utils import get_net
from tqdm import tqdm

def init_quant_model(model, labels, args):
    # Quantizer config
    wq_params = {'n_bits': args.w_bit, 'dim': 0, 'scale_method': ('max', 0)}
    aq_params = {'n_bits': args.a_bit, 'dim': (0, 1), 'scale_method': (args.scale_method, args.eps), 'leaf_param': True}
    matmul_params = {'n_bits': args.a_bit, 'dim': (0, 1, 2), 'scale_method': (args.scale_method, args.eps), 'leaf_param': True}

    qnn = QuantVAR(get_net(model), wq_params, aq_params, matmul_params)
    qnn.cuda()
    qnn.eval()

    device = next(qnn.parameters()).device

    qnn.get_rng_state()
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            _ = qnn.autoregressive_infer_cfg(B=labels[0].size(0), label_B=labels[0].to(device), cfg=args.cfg, top_k=900, top_p=0.96, g_seed=args.seed, more_smooth=False)
    qnn.set_rng_state()

    for module in qnn.modules():
        if isinstance(module, (QuantModule, QuantMatMul)):
            module.init_patch_res = True

    # Init quantizers
    qnn.set_quant_state(True, True)
    for module in qnn.modules():
        if isinstance(module, (UniformAffineQuantizer, GroupAffineQuantizer)):
            module.inited = False

    for module in qnn.model.vae_quant_proxy[0].modules():
        if isinstance(module, (UniformAffineQuantizer, GroupAffineQuantizer)):
            module.inited = False

    qnn.get_rng_state()
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            _ = qnn.autoregressive_infer_cfg(B=labels[0].size(0), label_B=labels[0].to(device), cfg=args.cfg, top_k=900, top_p=0.96, g_seed=args.seed + labels[0].item(), more_smooth=False)
    qnn.set_rng_state()

    for module in qnn.modules():
        if isinstance(module, (UniformAffineQuantizer, GroupAffineQuantizer)):
            module.inited = True

    for module in qnn.model.vae_quant_proxy[0].modules():
        if isinstance(module, (UniformAffineQuantizer, GroupAffineQuantizer)):
            module.inited = True

    return qnn