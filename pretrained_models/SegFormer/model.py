## Segformer
import torch
# Build a minimal SegFormer-B0 (Cityscapes) config directly in code
def build_segformer_b0_mmseg(num_classes=19):
    norm_cfg = dict(type='SyncBN', requires_grad=True)
    model_cfg = dict(
        type='EncoderDecoder',
        data_preprocessor=dict(     # we won't use the full test_step; we call backbone/head directly
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255),
        backbone=dict(              # MixVisionTransformer (MiT-B0)
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            strides=[4, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            out_indices=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'),
            drop_rate=0.0,
            init_cfg=None),
        decode_head=dict(           # SegFormerHead
            type='SegformerHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=128,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
    )
    model = MODELS.build(model_cfg)
    return model

@torch.no_grad()
def segformer_mmseg_logits(model, x):
    """
    x: float tensor in [0,1], shape [B,3,H,W] (your pipeline format).
    Returns: logits [B, num_classes, H, W] (no upsample done).
    """
    # mmseg configs usually normalize with ImageNet mean/std (in 0..255 space).
    mean = torch.tensor([123.675, 116.28, 103.53], device=x.device)[None, :, None, None] / 255.0
    std  = torch.tensor([58.395,  57.12,  57.375], device=x.device)[None, :, None, None] / 255.0
    # Convert to RGB normalized expected by backbone (data_preprocessor would do this normally)
    x = (x - mean) / std
    feats = model.backbone(x)
    logits = model.decode_head(feats)
    return logits

def load_segformer_mmseg_from_local(ckpt_path: str, device: torch.device, num_classes: int = 19):
    model = build_segformer_b0_mmseg(num_classes=num_classes).to(device)
    # ckpt is your local Kaggle .pth from mmseg
    _ = load_checkpoint(model, ckpt_path, map_location=device)
    model.eval()
    return model
