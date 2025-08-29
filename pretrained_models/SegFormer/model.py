# pretrained_models/SegFormer/model.py
import torch
from collections import OrderedDict
from .segformer_b0 import SegFormer_B0  # the minimal PyTorch class we defined

def load_segformer_local(
    variant_or_name: str,
    ckpt_path: str,
    device,
    num_classes: int = 19,
    strict: bool = False,
    verbose: bool = True,
):
    """
    Load SegFormer from a local .pth checkpoint (no HF, no mmseg runtime).
    Currently supports 'segformer_b0' (MiT-B0 backbone) that matches Cityscapes 19 classes.
    """
    if variant_or_name.lower() not in {"segformer_b0", "b0"}:
        raise NotImplementedError(f"Only 'segformer_b0' supported here, got '{variant_or_name}'")

    # 1) build model
    model = SegFormer_B0(num_classes=num_classes).to(device)

    # 2) read checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)  # mmseg style often uses 'state_dict'

    # 3) light key remap (decode head naming differences)
    remap = OrderedDict()
    for k, v in state.items():
        nk = k
        # common mmseg decode-head keys → our head module names
        nk = nk.replace("decode_head.linear_c1", "decode_head.proj_c1.0")
        nk = nk.replace("decode_head.linear_c2", "decode_head.proj_c2.0")
        nk = nk.replace("decode_head.linear_c3", "decode_head.proj_c3.0")
        nk = nk.replace("decode_head.linear_c4", "decode_head.proj_c4.0")
        nk = nk.replace("decode_head.linear_fuse", "decode_head.fuse.0")
        nk = nk.replace("decode_head.conv_seg",  "decode_head.cls")
        # keep backbone.* as-is; our backbone names mirror mmseg’s
        remap[nk] = v

    # 4) load
    msg = model.load_state_dict(remap, strict=strict)

    if verbose:
        missing = list(msg.missing_keys)
        unexpected = list(msg.unexpected_keys)
        print(f"[SegFormer] matched ~{len(remap)-len(unexpected)}/{len(remap)} keys | "
              f"missing {len(missing)} | unexpected {len(unexpected)} | strict={strict}")
        if missing:    print("  missing (head/backbone not found):", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected: print("  unexpected (unused in this class):", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.eval()
    return model

@torch.no_grad()
def segformer_logits(model, image_bchw):
    # returns logits at 1/4 resolution (SegFormer head output)
    return model(image_bchw)
