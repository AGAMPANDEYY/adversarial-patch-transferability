# pretrained_models/SegFormer/model.py
# Loader + builder that work with local .pth checkpoints (mmseg-style)

import torch
import torch.nn as nn
from .arch import SegFormer

def build_segformer_from_name(name: str, num_classes: int) -> nn.Module:
    """
    name: 'segformer_b0' | 'segformer_b2' | 'b0' | 'b2' | 'mit_b0' ...
    """
    name = name.lower()
    if "b0" in name:
        variant = "b0"
    elif "b1" in name:
        variant = "b1"
    elif "b2" in name:
        variant = "b2"
    elif "b3" in name:
        variant = "b3"
    elif "b4" in name:
        variant = "b4"
    elif "b5" in name:
        variant = "b5"
    else:
        raise ValueError(f"Unknown segformer variant in '{name}'")
    return SegFormer(variant=variant, num_classes=num_classes)

def _strip_prefix(sd, prefix):
    if all(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

def load_segformer_local(variant_or_name: str, ckpt_path: str, device, num_classes: int,
                         strict: bool = False, verbose: bool = True) -> nn.Module:
    """
    Build a SegFormer and load a LOCAL .pth checkpoint (like mmseg's).
    This tries to be compatible with mmseg naming:
      - expects keys under 'backbone.*' and 'decode_head.*' etc.
    """
    model = build_segformer_from_name(variant_or_name, num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt)

    # common wrappers to remove
    for p in ('module.', 'model.'):
        sd = _strip_prefix(sd, p)

    if strict:
        model.load_state_dict(sd, strict=True)
        if verbose:
            print("[SegFormer] strict=True load complete.")
        return model

    # safe-load: keep only matching keys & shapes
    msd = model.state_dict()
    inter = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    missing = [k for k in msd.keys() if k not in inter]
    unexpected = [k for k in sd.keys() if k not in msd]

    if verbose:
        print(f"[SegFormer] matched {len(inter)}/{len(msd)} keys | "
              f"missing {len(missing)} | unexpected {len(unexpected)}")

    model.load_state_dict(inter, strict=False)
    model.eval()
    return model

@torch.no_grad()
def segformer_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Unified forward → logits tensor [B, C, H/4, W/4].
    """
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        return out.get('logits', next(iter(out.values())))
    raise TypeError(f"Unexpected SegFormer output type: {type(out)}")
