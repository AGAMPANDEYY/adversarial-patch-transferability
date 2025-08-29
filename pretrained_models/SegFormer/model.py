# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from .arch import SegFormer_B

def load_segformer_local(variant_or_name: str,
                         ckpt_path: str,
                         device: torch.device,
                         num_classes: int = 19,
                         decoder_channels: int = 128,
                         strict: bool = False,
                         verbose: bool = True):
    """
    Load SegFormer (B0..B5) from a local .pth checkpoint produced by NVLabs/mmseg.
    """
    # Build model with mmseg-compatible names
    model = SegFormer_B(variant=variant_or_name.split('_')[-1] if 'segformer_' in variant_or_name else variant_or_name,
                        num_classes=num_classes,
                        decoder_channels=decoder_channels).to(device)
    model.eval()

    # Read checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    # strip common prefixes
    new_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith('module.'):
            k2 = k2[len('module.'):]
        # mmseg often uses 'backbone.' and 'decode_head.' already; keep them
        # optional extra nesting:
        if k2.startswith('model.'):
            k2 = k2[len('model.'):]
        # we don't support auxiliary_head -> drop
        if k2.startswith('auxiliary_head.'):
            continue
        new_state[k2] = v

    # try to load
    msg = model.load_state_dict(new_state, strict=strict)

    if verbose:
        missing = []
        unexpected = []
        if hasattr(msg, 'missing_keys'):   missing = msg.missing_keys
        if hasattr(msg, 'unexpected_keys'):unexpected = msg.unexpected_keys
        # count matched
        matched = sum(1 for k in new_state.keys() if k not in unexpected)
        print(f"[SegFormer] matched {matched}/{len(new_state)} keys | missing {len(missing)} | unexpected {len(unexpected)}")
        if missing:
            print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    return model

@torch.no_grad()
def segformer_logits(model: SegFormer_B, x: torch.Tensor):
    """
    Forward to 1/4-resolution logits. Upsample yourself if you need full-res.
    """
    return model(x)  # [B, num_classes, H/4, W/4]
