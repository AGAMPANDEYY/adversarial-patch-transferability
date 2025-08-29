# pretrained_models/SegFormer/arch.py
# Minimal SegFormer (MiT backbone + head) with mmseg-style names.
# Designed to load local mmseg checkpoints without installing mmseg/hf.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ------------------------------
# Helpers
# ------------------------------
class DWConv(nn.Module):
    """Depthwise conv used inside MLP."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        x = x.transpose(1, 2).view(x.shape[0], x.shape[2], H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., use_dwconv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features) if use_dwconv else None
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        if self.dwconv is not None:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """Overlap Patch Embedding (mmseg naming: patch_embed{1..4})"""
    def __init__(self, in_chans, embed_dim, patch_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, H', W'
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Attention(nn.Module):
    """Spatial-reduction attention (mmseg-style)."""
    def __init__(self, dim, num_heads, sr_ratio=1, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., sr_ratio=1,
                 drop=0., attn_drop=0., drop_path=0., use_dwconv=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio),
                       drop=drop, use_dwconv=use_dwconv)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DropPath(nn.Module):
    """Stochastic depth (Identity if p==0)."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ------------------------------
# MiT backbone (mmseg-style names)
# ------------------------------
class MixVisionTransformer(nn.Module):
    """mmseg-like MixVisionTransformer backbone with stages 1..4."""
    def __init__(self,
                 embed_dims: List[int],
                 depths: List[int],
                 num_heads: List[int],
                 sr_ratios: List[int],
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()

        # patch_embed1..4
        self.patch_embed1 = OverlapPatchEmbed(3, embed_dims[0], patch_size=7, stride=4, padding=3)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1)

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratio=4., sr_ratio=sr_ratios[0],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[0])
        ])
        cur += depths[0]
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratio=4., sr_ratio=sr_ratios[1],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[1])
        ])
        cur += depths[1]
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratio=4., sr_ratio=sr_ratios[2],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[2])
        ])
        cur += depths[2]
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratio=4., sr_ratio=sr_ratios[3],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).transpose(1, 2).reshape(B, -1, H, W)  # 1/4

        # Stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).transpose(1, 2).reshape(B, -1, H, W)  # 1/8

        # Stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).transpose(1, 2).reshape(B, -1, H, W)  # 1/16

        # Stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).transpose(1, 2).reshape(B, -1, H, W)  # 1/32

        return x1, x2, x3, x4

# ------------------------------
# Decode head (mmseg-style names)
# ------------------------------
class MLPHead(nn.Module):
    """1x1 conv as Linear layer over channel last format."""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)

class SegFormerHead(nn.Module):
    """mmseg-like head with linear_c1..c4 and linear_fuse + conv_seg."""
    def __init__(self, in_channels: List[int], embedding_dim: int, num_classes: int):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c1 = MLPHead(c1, embedding_dim)
        self.linear_c2 = MLPHead(c2, embedding_dim)
        self.linear_c3 = MLPHead(c3, embedding_dim)
        self.linear_c4 = MLPHead(c4, embedding_dim)

        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        # all features to 1/4 resolution (x1 spatial size)
        H, W = x1.shape[2], x1.shape[3]
        _c1 = self.linear_c1(x1)
        _c2 = F.interpolate(self.linear_c2(x2), size=(H, W), mode='bilinear', align_corners=True)
        _c3 = F.interpolate(self.linear_c3(x3), size=(H, W), mode='bilinear', align_corners=True)
        _c4 = F.interpolate(self.linear_c4(x4), size=(H, W), mode='bilinear', align_corners=True)

        _c = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        _c = self.linear_fuse(_c)
        _c = self.bn(_c)
        _c = self.relu(_c)
        _c = self.dropout(_c)
        logits = self.conv_seg(_c)  # 1/4 logits
        return logits

# ------------------------------
# Complete model (mmseg-style names)
# ------------------------------
class SegFormer(nn.Module):
    """
    Wrapper with mmseg-like attribute names:
      - self.backbone (MixVisionTransformer)
      - self.decode_head (SegFormerHead)
    Forward returns raw 1/4 logits; upsample outside to input size if needed.
    """
    def __init__(self, variant: str = "b0", num_classes: int = 19):
        super().__init__()
        variant = variant.lower()
        if variant in ("b0", "segformer_b0", "mit_b0"):
            embed_dims = [32, 64, 160, 256]
            depths     = [2, 2, 2, 2]
            num_heads  = [1, 2, 5, 8]
            sr_ratios  = [8, 4, 2, 1]
        elif variant in ("b1", "segformer_b1", "mit_b1"):
            embed_dims = [64, 128, 320, 512]
            depths     = [2, 2, 2, 2]
            num_heads  = [1, 2, 5, 8]
            sr_ratios  = [8, 4, 2, 1]
        elif variant in ("b2", "segformer_b2", "mit_b2"):
            embed_dims = [64, 128, 320, 512]
            depths     = [3, 4, 6, 3]
            num_heads  = [1, 2, 5, 8]
            sr_ratios  = [8, 4, 2, 1]
        elif variant in ("b3", "segformer_b3", "mit_b3"):
            embed_dims = [64, 128, 320, 512]
            depths     = [3, 4, 18, 3]
            num_heads  = [1, 2, 5, 8]
            sr_ratios  = [8, 4, 2, 1]
        elif variant in ("b4", "segformer_b4", "mit_b4"):
            embed_dims = [64, 128, 320, 512]
            depths     = [3, 8, 27, 3]
            num_heads  = [1, 2, 5, 8]
            sr_ratios  = [8, 4, 2, 1]
        elif variant in ("b5", "segformer_b5", "mit_b5"):
            embed_dims = [64, 128, 320, 512]
            depths     = [3, 6, 40, 3]
            num_heads  = [1, 2, 5, 8]
            sr_ratios  = [8, 4, 2, 1]
        else:
            raise ValueError(f"Unknown SegFormer variant: {variant}")

        self.backbone = MixVisionTransformer(
            embed_dims=embed_dims, depths=depths, num_heads=num_heads, sr_ratios=sr_ratios,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1
        )
        self.decode_head = SegFormerHead(in_channels=embed_dims, embedding_dim=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = self.backbone(x)
        logits = self.decode_head(x1, x2, x3, x4)  # 1/4 scale
        return logits
