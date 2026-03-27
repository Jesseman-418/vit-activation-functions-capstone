"""
Capstone Full Experiment Suite
===============================
5 Activation Functions x 4 Architecture Configs x 2 Datasets = 40 Experiments

Architectures:
  1. ViT (no SSL)   — trained from scratch
  2. ViT + DINO     — SSL pretrained → fine-tuned
  3. CaiT + DINO    — SSL pretrained → fine-tuned
  4. Swin + DINO    — SSL pretrained → fine-tuned

Activations: GELU, ReLU, SiLU, Tanh, Mish
Datasets: CIFAR-10, Biomedical

Run on Google Colab with GPU runtime.
"""

# ==============================================================================
# CELL 1: Setup & Imports
# ==============================================================================

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "colorama", "einops", "timm", "medmnist"])

import os
import math
import time
import json
import random
import logging
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_

warnings.filterwarnings('ignore')

# ==============================================================================
# CELL 2: Configuration
# ==============================================================================

# --- EDIT THESE PATHS ---
DRIVE_ROOT = "/content/gdrive/MyDrive"  # Google Drive mount point
PROJECT_DIR = os.path.join(DRIVE_ROOT, "Project")  # Where your project lives
BIOMEDICAL_ZIP = os.path.join(PROJECT_DIR, "datasets/biomedical.zip")
BIOMEDICAL_DIR = os.path.join(PROJECT_DIR, "datasets/biomedical")
CIFAR10_DIR = os.path.join(PROJECT_DIR, "data")  # CIFAR-10 data path
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# --- EXPERIMENT MATRIX ---
ACTIVATIONS = {
    "GELU": nn.GELU,
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
    "Mish": nn.Mish,
}

# 4 architecture configs: (arch, use_ssl)
ARCH_CONFIGS = [
    ("vit",  False, "ViT (no SSL)"),      # ViT trained from scratch
    ("vit",  True,  "ViT + DINO"),         # ViT with SSL pretraining
    ("cait", True,  "CaiT + DINO"),        # CaiT with SSL pretraining
    ("swin", True,  "Swin + DINO"),        # Swin with SSL pretraining
]

DATASETS = ["CIFAR10", "Biomedical"]

# --- TRAINING HYPERPARAMS ---
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 100
BATCH_SIZE = 256
PRETRAIN_LR = 0.0001
FINETUNE_LR = 0.001
WEIGHT_DECAY = 0.04
CLIP_GRAD = 3.0
WARMUP_EPOCHS = 30
FINETUNE_WARMUP = 10
SEED = 0

# --- CONTROL FLAGS ---
FAST_MODE = False       # Reduced epochs (pretrain=50, finetune=30) for testing
RUN_DATASETS = None     # Set to ["CIFAR10"] or ["Biomedical"] to run only one dataset, None = both
RUN_ARCHS = None        # Set to [0,1,2,3] indices to run specific configs, None = all

if FAST_MODE:
    PRETRAIN_EPOCHS = 50
    FINETUNE_EPOCHS = 30
    WARMUP_EPOCHS = 10
    FINETUNE_WARMUP = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ==============================================================================
# CELL 3: Dataset Setup (Biomedical)
# ==============================================================================

def setup_dataset(dataset_name):
    """Setup CIFAR-10 or Biomedical dataset. Returns (train_dir, val_dir, n_classes, img_size, img_mean, img_std)."""

    if dataset_name == "CIFAR10":
        # CIFAR-10: use torchvision (auto-downloads)
        n_classes = 10
        img_size = 32
        img_mean = (0.4914, 0.4822, 0.4465)
        img_std = (0.2470, 0.2435, 0.2616)
        # We'll handle CIFAR-10 specially in data loading (not ImageFolder)
        return None, None, n_classes, img_size, img_mean, img_std

    elif dataset_name == "Biomedical":
        # Try to load from Drive
        if not os.path.exists(BIOMEDICAL_DIR) and os.path.exists(BIOMEDICAL_ZIP):
            print(f"Extracting {BIOMEDICAL_ZIP}...")
            import zipfile
            with zipfile.ZipFile(BIOMEDICAL_ZIP, 'r') as z:
                z.extractall(os.path.dirname(BIOMEDICAL_DIR))
            print("Done.")

        if os.path.exists(BIOMEDICAL_DIR):
            if os.path.isdir(os.path.join(BIOMEDICAL_DIR, "train")):
                train_dir = os.path.join(BIOMEDICAL_DIR, "train")
                val_dir = os.path.join(BIOMEDICAL_DIR, "val") if os.path.isdir(os.path.join(BIOMEDICAL_DIR, "val")) else os.path.join(BIOMEDICAL_DIR, "test")
            else:
                train_dir = BIOMEDICAL_DIR
                val_dir = None

            classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            n_classes = len(classes)

            from PIL import Image
            for cls in classes:
                cls_dir = os.path.join(train_dir, cls)
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img = Image.open(os.path.join(cls_dir, img_name))
                        img_size = max(img.size)
                        img_size = max(32, ((img_size + 7) // 8) * 8)
                        print(f"Biomedical: {n_classes} classes, img_size={img_size}")
                        img_mean, img_std = compute_dataset_stats(train_dir, img_size)
                        return train_dir, val_dir, n_classes, img_size, img_mean, img_std

        # Fallback: MedMNIST PathMNIST (colorectal histology — 9 tissue types)
        print("Biomedical zip not found. Downloading PathMNIST from MedMNIST...")
        import medmnist
        from medmnist import PathMNIST

        data_dir = os.path.join(os.path.dirname(BIOMEDICAL_DIR), "pathmnist")
        os.makedirs(data_dir, exist_ok=True)

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(data_dir, split)
            if os.path.exists(split_dir):
                continue
            ds = PathMNIST(split=split, download=True, root=data_dir)
            for i in range(len(ds)):
                img, label = ds[i]
                label = int(label.item()) if hasattr(label, 'item') else int(label[0])
                cls_dir = os.path.join(split_dir, f"class_{label:02d}")
                os.makedirs(cls_dir, exist_ok=True)
                img.save(os.path.join(cls_dir, f"{i:06d}.png"))

        img_mean, img_std = compute_dataset_stats(os.path.join(data_dir, "train"), 32)
        return os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), 9, 32, img_mean, img_std

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_dataset_stats(train_dir, img_size):
    """Compute mean and std of the dataset."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=256, num_workers=2, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n = 0
    for imgs, _ in tqdm(loader, desc="Computing stats"):
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, 3, -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        n += batch_samples

    mean /= n
    std /= n
    return tuple(mean.tolist()), tuple(std.tolist())


# ==============================================================================
# CELL 4: Model Definitions (with act_layer support)
# ==============================================================================

# --- ViT ---

class VitMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VitAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VitBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VitAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = VitMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VitPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=9,
                 num_heads=12, mlp_ratio=2., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = VitPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     drop=drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))


# --- CaiT ---

class CaitClassAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q * self.scale @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class CaitTalkingHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CaitLayerScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CaitTalkingHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = VitMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CaitLayerScaleBlockCA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CaitClassAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = VitMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class CaiTModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=24,
                 num_heads=4, mlp_ratio=2., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_scale=1e-5, depth_token_only=2):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        patch_dim = in_chans * patch_size ** 2
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, embed_dim),
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.ModuleList([
            CaitLayerScaleBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                drop=drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer,
                                init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([
            CaitLayerScaleBlockCA(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=qkv_bias,
                                  drop=0., drop_path=0., act_layer=act_layer, norm_layer=norm_layer,
                                  init_values=init_scale)
            for _ in range(depth_token_only)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.pos_drop(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        for blk in self.blocks:
            x = blk(x)
        for blk in self.blocks_token_only:
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        return self.head(x[:, 0])


# --- Swin Transformer ---

class SwinMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = SwinWindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwinMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.attn_mask_dict = {}

    def create_attn_mask(self, H, W, device):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        cnt = 0
        for h in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
            for w in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if H not in self.attn_mask_dict:
                self.attn_mask_dict[H] = self.create_attn_mask(H, W, x.device)
            attn_mask = self.attn_mask_dict[H]
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class SwinPatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)


class SwinBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, act_layer=nn.GELU):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, num_classes=10,
                 embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],
                 window_size=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = SwinPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                          embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, act_layer=act_layer,
                downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return self.head(x)


# ==============================================================================
# CELL 5: Model Factory
# ==============================================================================

def create_model(arch, img_size, n_classes, act_layer=nn.GELU, drop_path_rate=0.1, mlp_ratio=2):
    patch_size_vit = 4 if img_size == 32 else 8
    patch_size_swin = 2 if img_size == 32 else 4

    if arch == "vit":
        return VisionTransformer(img_size=img_size, patch_size=patch_size_vit, num_classes=n_classes,
                                 embed_dim=192, depth=9, num_heads=12, mlp_ratio=mlp_ratio,
                                 drop_path_rate=drop_path_rate, act_layer=act_layer)
    elif arch == "cait":
        return CaiTModel(img_size=img_size, patch_size=patch_size_vit, num_classes=n_classes,
                         embed_dim=192, depth=24, num_heads=4, mlp_ratio=mlp_ratio,
                         drop_path_rate=drop_path_rate, act_layer=act_layer,
                         init_scale=1e-5, depth_token_only=2)
    elif arch == "swin":
        return SwinTransformer(img_size=img_size, patch_size=patch_size_swin, num_classes=n_classes,
                               embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],
                               window_size=4, mlp_ratio=mlp_ratio,
                               drop_path_rate=drop_path_rate, act_layer=act_layer)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ==============================================================================
# CELL 6: Training Utilities
# ==============================================================================

class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, max_lr=0.001, min_lr=1e-6,
                 warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr]
        else:
            return [self.min_lr + (self.max_lr - self.min_lr) *
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].flatten().float().sum(0) * 100.0 / batch_size for k in topk]


# ==============================================================================
# CELL 7: Training & Validation Functions
# ==============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, use_cutmix=True, use_mixup=True, mix_prob=0.5):
    model.train()
    total_loss, total_correct, total_samples = 0., 0, 0

    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # CutMix / Mixup
        r = np.random.rand()
        if r < mix_prob and use_cutmix:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size(0)).to(DEVICE)
            target_a, target_b = targets, targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
            output = model(images)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        elif r < mix_prob * 2 and use_mixup:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size(0)).to(DEVICE)
            mixed = lam * images + (1 - lam) * images[rand_index]
            target_a, target_b = targets, targets[rand_index]
            output = model(mixed)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(images)
            loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = output.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, 100. * total_correct / total_samples


@torch.no_grad()
def validate(model, val_loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0., 0, 0

    for images, targets in val_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        output = model(images)
        loss = criterion(output, targets)
        total_loss += loss.item() * images.size(0)
        _, predicted = output.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, 100. * total_correct / total_samples


# ==============================================================================
# CELL 8: DINO SSL Pretraining
# ==============================================================================

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, out_dim))
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(out_dim, out_dim, bias=False)) if norm_last_layer else nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class ViewPredLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_teacher_temp_epochs=10, nepochs=200, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach().chunk(2)
        total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone.forward_features(torch.cat(x[start_idx: end_idx]))
            output = torch.cat((output, _out))
            start_idx = end_idx
        return self.head(output)


def pretrain_dino(arch, act_layer, act_name, img_size, img_mean, img_std, train_dir, n_classes, dataset_name="Biomedical"):
    """Run DINO self-supervised pretraining for a given arch+activation combo."""

    exp_name = f"{dataset_name}_{arch}_{act_name}"
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Check for existing checkpoint
    final_ckpt = os.path.join(ckpt_dir, "pretrain_checkpoint.pth")
    if os.path.exists(final_ckpt):
        print(f"  [SKIP] Pretrain checkpoint exists: {final_ckpt}")
        return final_ckpt

    print(f"  Pretraining {arch}/{act_name} with DINO ({PRETRAIN_EPOCHS} epochs)...")

    # Data augmentation for DINO
    global_size = img_size
    local_size = img_size // 2
    normalize = transforms.Normalize(img_mean, img_std)

    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(global_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), normalize])

    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(local_size, scale=(0.2, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), normalize])

    class MultiCropDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, global_transform, local_transform, n_local=8):
            self.dataset = base_dataset
            self.global_t = global_transform
            self.local_t = local_transform
            self.n_local = n_local

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, _ = self.dataset[idx]
            crops = [self.global_t(img), self.global_t(img)]
            for _ in range(self.n_local):
                crops.append(self.local_t(img))
            return crops

    if dataset_name == "CIFAR10":
        base_dataset = datasets.CIFAR10(root=CIFAR10_DIR, train=True, download=True)
    else:
        base_dataset = datasets.ImageFolder(train_dir)

    dataset = MultiCropDataset(base_dataset, global_transform, local_transform, n_local=8)
    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, 128), shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True,
                        collate_fn=lambda batch: [torch.stack([b[i] for b in batch]) for i in range(10)])

    # Student and teacher
    student_backbone = create_model(arch, img_size, 0, act_layer=act_layer)
    teacher_backbone = create_model(arch, img_size, 0, act_layer=act_layer)
    embed_dim = student_backbone.embed_dim

    student = MultiCropWrapper(student_backbone, MLPHead(embed_dim, 1024)).to(DEVICE)
    teacher = MultiCropWrapper(teacher_backbone, MLPHead(embed_dim, 1024, norm_last_layer=False)).to(DEVICE)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Loss + optimizer
    dino_loss = ViewPredLoss(1024, 10, nepochs=PRETRAIN_EPOCHS).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)

    # Momentum schedule
    momentum_schedule = np.concatenate([
        np.linspace(0.996, 1.0, PRETRAIN_EPOCHS)])

    for epoch in range(PRETRAIN_EPOCHS):
        student.train()
        total_loss = 0.
        for crops in loader:
            crops = [c.to(DEVICE) for c in crops]
            student_out = student(crops)
            teacher_out = teacher(crops[:2])
            loss = dino_loss(student_out, teacher_out, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), CLIP_GRAD)
            optimizer.step()

            # EMA update teacher
            with torch.no_grad():
                m = momentum_schedule[epoch]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(m).add_((1 - m) * ps.data)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1}/{PRETRAIN_EPOCHS}] Loss: {avg_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % 50 == 0 or epoch == PRETRAIN_EPOCHS - 1:
            torch.save({
                'epoch': epoch,
                'teacher': teacher.state_dict(),
                'student': student.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, final_ckpt)

    print(f"  Pretrain done. Saved to {final_ckpt}")
    return final_ckpt


# ==============================================================================
# CELL 9: Fine-tuning Function
# ==============================================================================

def finetune(arch, act_layer, act_name, img_size, n_classes, img_mean, img_std,
             train_dir, val_dir, dataset_name, config_label, pretrain_ckpt=None):
    """Fine-tune a model with supervised learning."""

    exp_name = f"{dataset_name}_{config_label.replace(' ', '_').replace('+', '')}_{act_name}"
    save_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Check if already done
    results_file = os.path.join(save_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"  [SKIP] Already completed: {config_label}/{act_name}/{dataset_name} -> {results['best_val_acc']:.2f}%")
        return results

    print(f"  Fine-tuning {config_label}/{act_name} on {dataset_name} ({FINETUNE_EPOCHS} epochs)...")

    # Data
    normalize = [transforms.Normalize(img_mean, img_std)]
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(), *normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.4), ratio=(0.3, 3.3)),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), *normalize])

    # Load dataset
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=CIFAR10_DIR, train=True, download=True, transform=train_transforms)
        val_dataset = datasets.CIFAR10(root=CIFAR10_DIR, train=False, download=True, transform=val_transforms)
    else:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        if val_dir and os.path.exists(val_dir):
            val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        else:
            n = len(train_dataset)
            n_val = int(0.2 * n)
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [n - n_val, n_val],
                generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = create_model(arch, img_size, n_classes, act_layer=act_layer)

    # Load pretrained weights if available
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"    Loading pretrained weights from {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        state_dict = ckpt.get("teacher", ckpt)
        # Strip wrapper prefixes
        new_sd = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("backbone.", "")
            if k.startswith("head.") or k.startswith("mlp.") or k.startswith("last_layer"):
                continue  # Skip projection head
            new_sd[k] = v
        # Load with size mismatch handling
        model_sd = model.state_dict()
        for k in new_sd:
            if k in model_sd and new_sd[k].shape != model_sd[k].shape:
                new_sd[k] = model_sd[k]
        model.load_state_dict(new_sd, strict=False)

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters: {n_params:.2f}M")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    total_steps = FINETUNE_EPOCHS * len(train_loader)
    warmup_steps = FINETUNE_WARMUP * len(train_loader)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=total_steps,
                                              max_lr=FINETUNE_LR, min_lr=1e-6, warmup_steps=warmup_steps)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.

    for epoch in range(FINETUNE_EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
        v_loss, v_acc = validate(model, val_loader, criterion)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1}/{FINETUNE_EPOCHS}] "
                  f"Train: {t_acc:.2f}% | Val: {v_acc:.2f}% (best: {best_val_acc:.2f}%) | "
                  f"Loss: {v_loss:.4f}")

    # Save results
    results = {
        "arch": arch,
        "config_label": config_label,
        "activation": act_name,
        "dataset": dataset_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "n_params_M": n_params,
        "epochs": FINETUNE_EPOCHS,
        "pretrained": pretrain_ckpt is not None,
        "history": history,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save history CSV
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)

    print(f"  Done: {config_label}/{act_name}/{dataset_name} -> Best Val Acc: {best_val_acc:.2f}%")
    return results


# ==============================================================================
# CELL 10: Run Full Experiment Matrix
# ==============================================================================

def run_all_experiments():
    """Run all 40 experiments: 5 activations x 4 arch configs x 2 datasets."""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Mount Drive if on Colab
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
    except ImportError:
        print("Not on Colab, using local paths.")

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Determine which datasets and archs to run
    ds_list = RUN_DATASETS if RUN_DATASETS else DATASETS
    arch_indices = RUN_ARCHS if RUN_ARCHS else list(range(len(ARCH_CONFIGS)))

    all_results = []
    total = len(ds_list) * len(arch_indices) * len(ACTIVATIONS)
    current = 0

    for dataset_name in ds_list:
        print(f"\n{'#' * 70}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        # Setup dataset
        train_dir, val_dir, n_classes, img_size, img_mean, img_std = setup_dataset(dataset_name)
        print(f"  Classes: {n_classes}, Image size: {img_size}")
        print(f"  Mean: {img_mean}, Std: {img_std}")

        for arch_idx in arch_indices:
            arch, use_ssl, config_label = ARCH_CONFIGS[arch_idx]

            for act_name, act_layer in ACTIVATIONS.items():
                current += 1
                print(f"\n{'=' * 60}")
                print(f"EXPERIMENT [{current}/{total}]: {config_label} + {act_name} on {dataset_name}")
                print(f"{'=' * 60}")

                # Phase 1: Pretraining (skip for ViT no-SSL)
                pretrain_ckpt = None
                if use_ssl:
                    if dataset_name == "CIFAR10":
                        # For CIFAR-10 DINO, we need ImageFolder format
                        # Use torchvision CIFAR10 dataset directly in pretrain
                        pretrain_ckpt = pretrain_dino(arch, act_layer, act_name,
                                                     img_size, img_mean, img_std,
                                                     train_dir, n_classes,
                                                     dataset_name=dataset_name)
                    else:
                        pretrain_ckpt = pretrain_dino(arch, act_layer, act_name,
                                                     img_size, img_mean, img_std,
                                                     train_dir, n_classes,
                                                     dataset_name=dataset_name)

                # Phase 2: Fine-tuning
                results = finetune(arch, act_layer, act_name, img_size, n_classes,
                                  img_mean, img_std, train_dir, val_dir,
                                  dataset_name, config_label, pretrain_ckpt)
                all_results.append(results)

    # ===== FINAL SUMMARY =====
    print(f"\n{'#' * 70}")
    print("FINAL RESULTS — All Experiments")
    print(f"{'#' * 70}")

    summary = pd.DataFrame([{
        "Dataset": r["dataset"],
        "Architecture": r["config_label"],
        "Activation": r["activation"],
        "Best Val Acc (%)": f"{r['best_val_acc']:.2f}",
        "Final Val Loss": f"{r['final_val_loss']:.4f}",
        "Params (M)": f"{r['n_params_M']:.2f}",
        "SSL Pretrained": r["pretrained"],
    } for r in all_results])

    # Print per-dataset summary
    for ds in ds_list:
        ds_summary = summary[summary["Dataset"] == ds]
        print(f"\n--- {ds} ---")
        print(ds_summary.to_string(index=False))

    # Save
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_all.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to {RESULTS_DIR}")
    return all_results


# ==============================================================================
# CELL 11: Visualization
# ==============================================================================

def plot_results(results_dir=RESULTS_DIR):
    """Generate comprehensive comparison plots from saved results."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 150

    results_file = os.path.join(results_dir, "all_results.json")
    with open(results_file) as f:
        all_results = json.load(f)

    acts = list(ACTIVATIONS.keys())
    config_labels = [c[2] for c in ARCH_CONFIGS]
    ds_list = sorted(set(r["dataset"] for r in all_results))
    colors = {'GELU': '#2196F3', 'ReLU': '#FF5722', 'SiLU': '#4CAF50', 'Tanh': '#FF9800', 'Mish': '#9C27B0'}

    # =========================================================================
    # PLOT 1: Per-dataset bar charts (4 architectures, 5 activations each)
    # =========================================================================
    for ds in ds_list:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        configs_in_ds = sorted(set(r["config_label"] for r in ds_results),
                              key=lambda x: config_labels.index(x) if x in config_labels else 99)

        fig, axes = plt.subplots(1, len(configs_in_ds), figsize=(5 * len(configs_in_ds), 6), sharey=True)
        if len(configs_in_ds) == 1:
            axes = [axes]

        for idx, cfg in enumerate(configs_in_ds):
            cfg_results = sorted([r for r in ds_results if r["config_label"] == cfg],
                                key=lambda r: acts.index(r["activation"]) if r["activation"] in acts else 99)
            accs = [r["best_val_acc"] for r in cfg_results]
            act_names = [r["activation"] for r in cfg_results]
            bar_colors = [colors.get(a, '#888') for a in act_names]
            bars = axes[idx].bar(act_names, accs, color=bar_colors, edgecolor='black', linewidth=0.5)
            axes[idx].set_title(cfg, fontsize=13, fontweight='bold')
            axes[idx].set_ylabel("Best Val Accuracy (%)" if idx == 0 else "")
            if accs:
                axes[idx].set_ylim(min(accs) - 5, max(accs) + 3)
            for bar, acc in zip(bars, accs):
                axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                              f'{acc:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=30)

        plt.suptitle(f"{ds} — Best Validation Accuracy by Architecture & Activation",
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{ds}_accuracy_bars.png"), dpi=150, bbox_inches='tight')
        plt.show()

    # =========================================================================
    # PLOT 2: Training convergence curves (per dataset, per architecture)
    # =========================================================================
    for ds in ds_list:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        configs_in_ds = sorted(set(r["config_label"] for r in ds_results),
                              key=lambda x: config_labels.index(x) if x in config_labels else 99)

        fig, axes = plt.subplots(1, len(configs_in_ds), figsize=(5 * len(configs_in_ds), 5), sharey=True)
        if len(configs_in_ds) == 1:
            axes = [axes]

        for idx, cfg in enumerate(configs_in_ds):
            cfg_results = sorted([r for r in ds_results if r["config_label"] == cfg],
                                key=lambda r: acts.index(r["activation"]) if r["activation"] in acts else 99)
            for r in cfg_results:
                if "history" in r and r["history"]:
                    axes[idx].plot(r["history"]["val_acc"], label=r["activation"],
                                  color=colors.get(r["activation"], '#888'), linewidth=1.5)
            axes[idx].set_title(cfg, fontsize=13, fontweight='bold')
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Val Accuracy (%)" if idx == 0 else "")
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f"{ds} — Validation Accuracy Convergence", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{ds}_convergence.png"), dpi=150, bbox_inches='tight')
        plt.show()

    # =========================================================================
    # PLOT 3: Cross-dataset heatmaps (one per architecture config)
    # =========================================================================
    for ds in ds_list:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        configs_in_ds = sorted(set(r["config_label"] for r in ds_results),
                              key=lambda x: config_labels.index(x) if x in config_labels else 99)
        acts_in_ds = sorted(set(r["activation"] for r in ds_results),
                           key=lambda x: acts.index(x) if x in acts else 99)

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(configs_in_ds)), max(5, len(acts_in_ds))))
        data = np.zeros((len(acts_in_ds), len(configs_in_ds)))
        for r in ds_results:
            if r["activation"] in acts_in_ds and r["config_label"] in configs_in_ds:
                i = acts_in_ds.index(r["activation"])
                j = configs_in_ds.index(r["config_label"])
                data[i, j] = r["best_val_acc"]

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=data[data > 0].min() - 2 if (data > 0).any() else 0)
        ax.set_xticks(range(len(configs_in_ds)))
        ax.set_xticklabels(configs_in_ds, fontsize=10)
        ax.set_yticks(range(len(acts_in_ds)))
        ax.set_yticklabels(acts_in_ds, fontsize=10)
        for i in range(len(acts_in_ds)):
            for j in range(len(configs_in_ds)):
                ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center", fontsize=11, fontweight='bold',
                       color='white' if data[i,j] < np.median(data) else 'black')
        plt.colorbar(im, label="Best Val Accuracy (%)")
        plt.title(f"{ds} — Activation x Architecture Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{ds}_heatmap.png"), dpi=150, bbox_inches='tight')
        plt.show()

    # =========================================================================
    # PLOT 4: Cross-dataset comparison (side-by-side for each architecture)
    # =========================================================================
    if len(ds_list) >= 2:
        all_configs = sorted(set(r["config_label"] for r in all_results),
                            key=lambda x: config_labels.index(x) if x in config_labels else 99)

        fig, axes = plt.subplots(1, len(all_configs), figsize=(5 * len(all_configs), 6), sharey=True)
        if len(all_configs) == 1:
            axes = [axes]

        x = np.arange(len(acts))
        width = 0.35

        for idx, cfg in enumerate(all_configs):
            for d_idx, ds in enumerate(ds_list):
                cfg_ds_results = sorted([r for r in all_results if r["config_label"] == cfg and r["dataset"] == ds],
                                       key=lambda r: acts.index(r["activation"]) if r["activation"] in acts else 99)
                accs = [r["best_val_acc"] for r in cfg_ds_results]
                act_labels = [r["activation"] for r in cfg_ds_results]
                if accs:
                    offset = (d_idx - 0.5) * width
                    bars = axes[idx].bar(x[:len(accs)] + offset, accs, width, label=ds, alpha=0.85,
                                        edgecolor='black', linewidth=0.5)
                    for bar, acc in zip(bars, accs):
                        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                                      f'{acc:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

            axes[idx].set_title(cfg, fontsize=13, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(acts, fontsize=9, rotation=30)
            axes[idx].set_ylabel("Best Val Accuracy (%)" if idx == 0 else "")
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.2, axis='y')

        plt.suptitle("Cross-Dataset Comparison — CIFAR-10 vs Biomedical", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "cross_dataset_comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()

    # =========================================================================
    # PLOT 5: SSL Impact (ViT no-SSL vs ViT+DINO)
    # =========================================================================
    for ds in ds_list:
        vit_nosssl = [r for r in all_results if r["dataset"] == ds and r["config_label"] == "ViT (no SSL)"]
        vit_ssl = [r for r in all_results if r["dataset"] == ds and r["config_label"] == "ViT + DINO"]

        if vit_nosssl and vit_ssl:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(acts))
            width = 0.35

            no_ssl_accs = {r["activation"]: r["best_val_acc"] for r in vit_nosssl}
            ssl_accs = {r["activation"]: r["best_val_acc"] for r in vit_ssl}

            no_ssl_vals = [no_ssl_accs.get(a, 0) for a in acts]
            ssl_vals = [ssl_accs.get(a, 0) for a in acts]
            gains = [ssl_vals[i] - no_ssl_vals[i] for i in range(len(acts))]

            bars1 = ax.bar(x - width/2, no_ssl_vals, width, label='ViT (no SSL)', color='#E57373', edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, ssl_vals, width, label='ViT + DINO', color='#42A5F5', edgecolor='black', linewidth=0.5)

            for i, (b1, b2, gain) in enumerate(zip(bars1, bars2, gains)):
                ax.text(b1.get_x() + b1.get_width()/2., b1.get_height() + 0.2,
                       f'{no_ssl_vals[i]:.1f}', ha='center', va='bottom', fontsize=9)
                ax.text(b2.get_x() + b2.get_width()/2., b2.get_height() + 0.2,
                       f'{ssl_vals[i]:.1f}', ha='center', va='bottom', fontsize=9)
                ax.annotate(f'+{gain:.1f}%', xy=(x[i] + width/2, ssl_vals[i]),
                           xytext=(x[i] + width/2 + 0.15, ssl_vals[i] + 1.5),
                           fontsize=8, color='green', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1))

            ax.set_xlabel("Activation Function")
            ax.set_ylabel("Best Validation Accuracy (%)")
            ax.set_title(f"{ds} — Impact of DINO Self-Supervised Pretraining on ViT", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(acts)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.2, axis='y')
            avg_gain = np.mean(gains)
            ax.text(0.98, 0.02, f'Avg SSL Gain: +{avg_gain:.2f}%', transform=ax.transAxes,
                   ha='right', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{ds}_ssl_impact.png"), dpi=150, bbox_inches='tight')
            plt.show()

    # =========================================================================
    # PLOT 6: Best activation per architecture (winner table)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BEST ACTIVATION PER ARCHITECTURE PER DATASET")
    print("=" * 70)

    winner_data = []
    for ds in ds_list:
        for cfg in config_labels:
            cfg_results = [r for r in all_results if r["dataset"] == ds and r["config_label"] == cfg]
            if cfg_results:
                best = max(cfg_results, key=lambda r: r["best_val_acc"])
                winner_data.append({
                    "Dataset": ds,
                    "Architecture": cfg,
                    "Best Activation": best["activation"],
                    "Accuracy (%)": f"{best['best_val_acc']:.2f}",
                    "Runner-up": sorted(cfg_results, key=lambda r: -r["best_val_acc"])[1]["activation"] if len(cfg_results) > 1 else "N/A",
                })

    winner_df = pd.DataFrame(winner_data)
    print(winner_df.to_string(index=False))
    winner_df.to_csv(os.path.join(results_dir, "winners.csv"), index=False)

    print(f"\nAll plots saved to {results_dir}/")


# ==============================================================================
# CELL 12: MAIN — Run Everything
# ==============================================================================

if __name__ == "__main__":
    results = run_all_experiments()
    plot_results()
