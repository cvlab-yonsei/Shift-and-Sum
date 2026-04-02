# Pytorch Implementation of Shift-and-Sum Quantization
This repository contains the official implementation of the paper "Shift-and-Sum Quantization for Visual Autoregressive Models" presented at **ICLR 2026**.

For detailed information, please visit the [project website](https://cvlab.yonsei.ac.kr/projects/Shift-and-Sum/) or read the [paper](https://iclr.cc/virtual/2026/poster/10010803).

## Getting started

### Installation
- Install required packages: `pip install -r requirements.txt`

### Pretrained models
- Download pretrained VAR models from the official VAR repository:
https://github.com/FoundationVision/VAR.

### Image Generation
- Generate 50,000 images using the following scripts. You may modify the arguments depending on your setup.
```bash
# Example code for quantizing VAR-d16 (6-bit quantization)
python main.py --n_samples=50000 --weight_dir /path/to/weight --save_dir /path/to/gen_imgs --w_bit 6 --a_bit 6 --model_depth 16 --lr 1e-4 --weight 1e-5 --channel_wise --head_wise

# Example code for quantizing VAR-d30 (4-bit quantization)
python main.py --n_samples=50000 --weight_dir /path/to/weight --save_dir /path/to/gen_imgs --w_bit 4 --a_bit 4 --model_depth 30 --lr 1e-4 --weight 1e-5 --channel_wise --head_wise
```

### Evaluation
- Evaluate **Inception Score (IS)** and **Fréchet inception distance (FID)** using OpenAI's evaluation toolkit:
https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py.

## Acknowledgement
The codebase of this repository is largely borrowed from [VAR](https://github.com/FoundationVision/VAR) and [BRECQ](https://github.com/yhhhli/BRECQ).
We sincerely thank the authors and contributors of these projects.

## Citation
If you find our work useful in your research, please cite our paper:
```
@inproceedings{moonshift,
  title={Shift-and-Sum Quantization for Visual Autoregressive Models},
  author={Moon, Jaehyeon and Ham, Bumsub},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```