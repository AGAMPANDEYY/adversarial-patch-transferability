# pretrained_models/SegFormer/segformer_b0.py
# Minimal SegFormer (MiT-B0) + head in pure PyTorch, names chosen to match mmseg-style keys as much as possible.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utils
# ------------------------------
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    def forward(self, x):
        return self.dwconv(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = self.proj(x)                       # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # B, N, C
        x = self.norm(x)
        return x, H, W

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden, drop=drop)
        self.drop_path = nn.Identity()  # keep simple; mmseg uses DropPath
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# ------------------------------
# MiT Backbone (B0)
# ------------------------------
class MixVisionTransformer_B0(nn.Module):
    def __init__(self, in_chans=3, embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2), num_heads=(1, 2, 5, 8)):
        super().__init__()
        self.embed_dims = embed_dims
        # Patch embeddings (names chosen to mirror mmseg's "backbone.patch_embed{1..4}")
        self.patch_embed1 = OverlapPatchEmbed(in_chans, embed_dims[0], patch_size=7, stride=4, padding=3)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1)
        # Blocks (names chosen to mirror mmseg's "backbone.block{1..4}.{i}")
        self.block1 = nn.ModuleList([TransformerBlock(embed_dims[0], num_heads[0]) for _ in range(depths[0])])
        self.norm1  = nn.LayerNorm(embed_dims[0], eps=1e-6)
        self.block2 = nn.ModuleList([TransformerBlock(embed_dims[1], num_heads[1]) for _ in range(depths[1])])
        self.norm2  = nn.LayerNorm(embed_dims[1], eps=1e-6)
        self.block3 = nn.ModuleList([TransformerBlock(embed_dims[2], num_heads[2]) for _ in range(depths[2])])
        self.norm3  = nn.LayerNorm(embed_dims[2], eps=1e-6)
        self.block4 = nn.ModuleList([TransformerBlock(embed_dims[3], num_heads[3]) for _ in range(depths[3])])
        self.norm4  = nn.LayerNorm(embed_dims[3], eps=1e-6)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        x1 = self.norm1(x).transpose(1, 2).reshape(B, self.embed_dims[0], H, W); outs.append(x1)

        x, H, W = self.patch_embed2(x1)
        for blk in self.block2: x = blk(x, H, W)
        x2 = self.norm2(x).transpose(1, 2).reshape(B, self.embed_dims[1], H, W); outs.append(x2)

        x, H, W = self.patch_embed3(x2)
        for blk in self.block3: x = blk(x, H, W)
        x3 = self.norm3(x).transpose(1, 2).reshape(B, self.embed_dims[2], H, W); outs.append(x3)

        x, H, W = self.patch_embed4(x3)
        for blk in self.block4: x = blk(x, H, W)
        x4 = self.norm4(x).transpose(1, 2).reshape(B, self.embed_dims[3], H, W); outs.append(x4)

        return outs  # [c1,c2,c3,c4]

# ------------------------------
# SegFormer decode head
# ------------------------------
class SegFormerHead(nn.Module):
    # mirrors mmseg decode_head with linear_c1..c4 + linear_fuse + classifier
    def __init__(self, in_channels=(32,64,160,256), embed_dim=256, num_classes=19, dropout=0.1):
        super().__init__()
        self.proj_c1 = nn.Sequential(
            nn.Conv2d(in_channels[0], embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(in_channels[1], embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj_c3 = nn.Sequential(
            nn.Conv2d(in_channels[2], embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj_c4 = nn.Sequential(
            nn.Conv2d(in_channels[3], embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.drop  = nn.Dropout2d(dropout)
        self.cls   = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats):
        c1, c2, c3, c4 = feats  # 1/4, 1/8, 1/16, 1/32
        H, W = c1.shape[-2:]
        c4 = F.interpolate(self.proj_c4(c4), size=(H,W), mode='bilinear', align_corners=False)
        c3 = F.interpolate(self.proj_c3(c3), size=(H,W), mode='bilinear', align_corners=False)
        c2 = F.interpolate(self.proj_c2(c2), size=(H,W), mode='bilinear', align_corners=False)
        c1 = self.proj_c1(c1)
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.fuse(x)
        x = self.drop(x)
        x = self.cls(x)     # B, num_classes, H/4, W/4  (if input was H,W)
        return x

# ------------------------------
# Full model
# ------------------------------
class SegFormer_B0(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.backbone = MixVisionTransformer_B0()
        self.decode_head = SegFormerHead(in_channels=(32,64,160,256), embed_dim=256, num_classes=num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        logits_1_4 = self.decode_head(feats)
        return logits_1_4
