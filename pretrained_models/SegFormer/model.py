# segformer_local_loader.py
import torch

def _strip_prefix(sd, prefix):
    if all(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

def load_segformer_local(build_fn, ckpt_path, device, strict=False, verbose=True):
    """
    Build your SegFormer model (from your repo), load local .pth checkpoint safely,
    and return it in eval mode.

    Args:
      build_fn: callable -> nn.Module  (e.g., lambda: SegFormerB0(num_classes=19))
      ckpt_path: str, local .pth
      device: torch.device
      strict: True to require exact key match; False = load intersection only
      verbose: print load stats
    """
    model = build_fn().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt)            # accept both raw dict or {"state_dict": ...}
    sd = _strip_prefix(sd, 'module:')            # typo guard
    sd = _strip_prefix(sd, 'module.')            # DP/DDP
    # (do NOT strip 'backbone.' / 'decode_head.' if your class uses mmseg-style names)

    if strict:
        model.load_state_dict(sd, strict=True)
    else:
        model_sd = model.state_dict()
        inter = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        miss  = [k for k in model_sd.keys() if k not in inter]
        unexp = [k for k in sd.keys() if k not in inter]
        if verbose:
            print(f"[SegFormer] safe-load: matched={len(inter)}/{len(model_sd)} "
                  f"| unexpected(ignored)={len(unexp)} | missing(left init)={len(miss)}")
        model.load_state_dict(inter, strict=False)

    model.eval()
    return model
