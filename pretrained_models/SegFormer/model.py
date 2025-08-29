# pretrained_models/SegFormer/model.py
# Local loader that tries to align an mmseg-style checkpoint with the minimal class above.

import torch
from collections import OrderedDict
from .segformer_b0 import SegFormer_B0

def _strip_prefix_if_present(state_dict, prefix):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }

def load_segformer_local(ckpt_path: str, device, num_classes=19):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)  # mmseg often stores under "state_dict"

    # mmseg usually prefixes with "backbone." and "decode_head."
    # our class uses the same names, so we mostly keep them.
    # sometimes keys are nested or have unused aux heads -> ignore by strict=False.
    model = SegFormer_B0(num_classes=num_classes).to(device)

    # OPTIONAL: small key mapping examples (expand if your checkpoint uses different naming)
    remap = OrderedDict()
    for k, v in state.items():
        nk = k
        # example: some checkpoints use "decode_head.linear_c1" etc. Our proj names differ slightly:
        nk = nk.replace("decode_head.linear_c1", "decode_head.proj_c1.0")
        nk = nk.replace("decode_head.linear_c2", "decode_head.proj_c2.0")
        nk = nk.replace("decode_head.linear_c3", "decode_head.proj_c3.0")
        nk = nk.replace("decode_head.linear_c4", "decode_head.proj_c4.0")
        nk = nk.replace("decode_head.linear_fuse", "decode_head.fuse.0")
        nk = nk.replace("decode_head.conv_seg", "decode_head.cls")
        # some checkpoints use "norm" vs "bn" in head (we used BN). Those layers won’t match — harmless.
        remap[nk] = v

    msg = model.load_state_dict(remap, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    print(f"[SegFormer] matched {len(remap)-len(unexpected)}/{len(remap)} keys | "
          f"missing {len(missing)} | unexpected {len(unexpected)}")
    if missing:
        print("  missing:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("  unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    model.eval()
    return model

@torch.no_grad()
def segformer_logits(model, image_bchw):
    """
    image_bchw: [B,3,H,W], normalized like your other models.
    Returns logits at 1/4 resolution (SegFormer head output). Up-sample yourself if needed.
    """
    return model(image_bchw)

