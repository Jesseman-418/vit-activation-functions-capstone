# Impact of Activation Functions on Vision Transformers for Image Classification

**Capstone Project** | VIT Chennai | 2026

## Authors
- **Jesseman Devamirtham N** (22BRS1112) - Lead
- R. Ritish Reddy (22BRS1291)
- Preeti Chanda Patra (22BRS1066)

**Guide:** Prof. Prasad M

## Abstract

A systematic study comparing the impact of different activation functions (GELU, ReLU, SiLU/Swish, Tanh, Mish) on Vision Transformer architectures (ViT, CaiT, Swin Transformer) for image classification. Uses a two-phase training pipeline: DINO self-supervised pretraining followed by supervised fine-tuning with advanced data augmentation.

## Key Findings (ViT on CIFAR-10)

| Activation | Best Val Accuracy | Rank |
|-----------|------------------|------|
| GELU | 93.68% | 1st |
| SiLU (Swish) | 93.12% | 2nd |
| ReLU | 92.71% | 3rd |
| Mish | 91.19% | 4th |
| Tanh | 88.50% | 5th |

- Non-monotonic activations (GELU, SiLU) outperform monotonic ones (ReLU, Tanh) by 1-5%
- DINO self-supervised pretraining provides +1.44% average accuracy gain across all activations
- Tanh benefits most from SSL pretraining (+2.00%)

## Project Structure

```
.
├── ViT/                          # Vision Transformer experiments
│   ├── models/                   # Model architectures with different activations
│   │   ├── vit.py               # Base ViT (GELU)
│   │   ├── vitReLU.py           # ViT with ReLU
│   │   ├── vitELU.py            # ViT with ELU
│   │   ├── vitSiLU.py           # ViT with SiLU/Swish
│   │   ├── vitTanh.py           # ViT with Tanh
│   │   ├── cait.py              # CaiT architecture
│   │   ├── caitELU.py           # CaiT with ELU
│   │   ├── caitSiLU.py          # CaiT with SiLU
│   │   ├── swin.py              # Swin Transformer
│   │   ├── swinELU.py           # Swin with ELU
│   │   ├── swinSiLU.py          # Swin with SiLU
│   │   └── build_model.py       # Unified model factory
│   ├── utils/                    # Training utilities
│   │   ├── autoaug.py           # AutoAugment (CIFAR10 policy)
│   │   ├── cosine_annealing_with_warmup.py
│   │   ├── dataloader.py        # Data loading & transforms
│   │   ├── drop_path.py         # Stochastic depth
│   │   ├── load_checkpoint.py   # Checkpoint loading
│   │   ├── logger_dict.py       # Metric logging
│   │   ├── losses.py            # Label smoothing CE loss
│   │   ├── mix.py               # CutMix & Mixup
│   │   ├── print_progress.py    # Training progress display
│   │   ├── projection_head.py   # DINO projection head
│   │   ├── random_erasing.py    # Random erasing augmentation
│   │   ├── sampler.py           # RASampler (repeated augmentation)
│   │   ├── scheduler.py         # LR scheduler
│   │   ├── training_functions.py # Train/eval loops
│   │   ├── transforms.py        # Data transforms
│   │   └── utils_ssl.py         # SSL utilities (DINO)
│   ├── pretraining_vit.ipynb    # Phase 1: DINO SSL pretraining
│   ├── finetune.ipynb           # Phase 2: Supervised fine-tuning
│   └── mix.py                   # CutMix/Mixup utilities
│
├── CaiT/                         # CaiT-specific experiments
│   ├── models/                   # Extended activation variants
│   │   ├── cait.py, vit.py, swin.py
│   │   ├── vitReLU.py, vitELU.py, vitTanh.py
│   │   ├── vitLeakyReLU.py, vitParametricReLU.py
│   │   ├── vitSigmoid.py, vitSwish.py
│   │   └── build_model.py
│   ├── utils/                    # Same utility suite
│   ├── pretraining_vit.ipynb
│   └── finetune.ipynb
│
└── Swin/                         # Swin Transformer experiments (WIP)
```

## Training Pipeline

### Phase 1: Self-Supervised Pretraining (DINO)
- Teacher-Student framework with EMA (momentum: 0.996 -> 1.0)
- Multi-crop strategy: 2 global + 8 local crops
- 200 epochs, AdamW (lr=0.0001), cosine schedule with 30-epoch warmup

### Phase 2: Supervised Fine-tuning
- SSL pretrained weight initialization
- Label Smoothing Cross-Entropy loss
- Advanced augmentation: AutoAugment + CutMix + Mixup + Random Erasing
- 100 epochs, AdamW (lr=0.001), CosineAnnealingWarmupRestarts
- Regularization: Stochastic Depth (0.1), Weight Decay (0.04), Gradient Clipping (3.0)

## Architecture Configurations

| Property | ViT | CaiT | Swin |
|----------|-----|------|------|
| Embedding Dim | 192 | 192 | 96 |
| Depth | 9 | 24 | [2, 6, 4] |
| Attention Heads | 12 | 4 | [3, 6, 12] |
| MLP Ratio | 2 | 2 | 2 |
| Patch Size | 4x4 | 4x4 | 2x2 |
| Parameters | ~2.7M | ~2.8M | ~3.1M |

## Datasets
- **CIFAR-10** (completed)
- CIFAR-100 (planned)
- Tiny-ImageNet (planned)
- Biomedical dataset (planned)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm
- tensorboard

## References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
2. Touvron et al., "Going Deeper with Image Transformers (CaiT)" (ICCV 2021)
3. Liu et al., "Swin Transformer" (ICCV 2021)
4. Caron et al., "DINO" (ICCV 2021)
5. Ramachandran et al., "Searching for Activation Functions" (2017)
6. Misra, "Mish: A Self Regularized Non-Monotonic Activation" (BMVC 2020)
7. Yun et al., "CutMix" (ICCV 2019)
8. Zhang et al., "mixup" (ICLR 2018)
