import sys, time, datetime, random
from typing import List, Tuple

# Keep your repo on sys.path for dataset/metrics/patch utils
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from dataset.cityscapes import Cityscapes
from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss           # uses your Stage-1 / Stage-2 logic
from patch.create import Patch               # for apply_patch
from torch.optim.lr_scheduler import ExponentialLR

# Hugging Face SegFormer (ViT-backbone)
from transformers import AutoConfig, SegformerForSemanticSegmentation

# Restore original sys.path to avoid other imports being shadowed
sys.path = original_sys_path


class PatchTrainerAttentionHijack:
    """
    Train an adversarial patch against a ViT-backbone segmenter (SegFormer)
    with an attention-hijack objective, then evaluate transfer to CNNs externally.

    Core ideas:
      - Patch is trainable (tanh-param for stability) OR PGD-updated (config switch)
      - Loss = (-attack_loss) + tv_weight * TV + attn_w * AttnHijack
      - AttnHijack: maximize attention mass TO patch tokens across layers/heads
      - EOT on the patch (affine + photometric jitter)
    """

    def __init__(self, config, main_logger):
        self.cfg = config
        self.log = main_logger
        self.device = config.experiment.device

        # -----------------------------
        # Dataloaders
        # -----------------------------
        cityscape_train = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.train,
            num_classes=config.dataset.num_classes,
            multi_scale=config.train.multi_scale,
            flip=config.train.flip,
            ignore_label=config.train.ignore_label,
            base_size=config.train.base_size,
            crop_size=(config.train.height, config.train.width),
            scale_factor=config.train.scale_factor,
        )
        cityscape_val = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.val,
            num_classes=config.dataset.num_classes,
            multi_scale=False,
            flip=False,
            ignore_label=config.train.ignore_label,
            base_size=config.test.base_size,
            crop_size=(config.test.height, config.test.width),
        )

        self.train_dl = torch.utils.data.DataLoader(
            cityscape_train,
            batch_size=config.train.batch_size,
            shuffle=config.train.shuffle,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            drop_last=config.train.drop_last,
        )
        self.val_dl = torch.utils.data.DataLoader(
            cityscape_val,
            batch_size=config.test.batch_size,
            shuffle=False,
            num_workers=config.test.num_workers,
            pin_memory=config.test.pin_memory,
            drop_last=config.test.drop_last,
        )

        self.iters_per_epoch = len(self.train_dl)
        self.start_epoch = config.train.start_epoch
        self.end_epoch   = config.train.end_epoch
        self.total_epochs = self.end_epoch - self.start_epoch
        assert self.total_epochs > 0

        # -----------------------------
        # Target model: HuggingFace SegFormer (ViT backbone)
        # -----------------------------
        hf_name = getattr(config.model, "hf_name", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        hf_cfg = AutoConfig.from_pretrained(
            hf_name,
            num_labels=config.dataset.num_classes,
            output_attentions=True,            # << crucial for attention hijack
            output_hidden_states=False,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(hf_name, config=hf_cfg)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)            # freeze model (we only train the patch)

        # -----------------------------
        # Losses, patch, optimizer
        # -----------------------------
        self.criterion = PatchLoss(config, main_logger)

        self.S = config.patch.size  # patch size SxS
        self.use_pgd = bool(getattr(config.optimizer, "use_pgd", False))
        self.lr = config.optimizer.init_lr

        # tanh-parameterized patch (stable); usable with Adam OR PGD
        self.patch_param = self.init_lowfreq_tanh((3, self.S, self.S), cutoff=0.2).to(self.device)
        # Optimizer if using Adam route
        if not self.use_pgd:
            self.opt = torch.optim.Adam([self.patch_param], lr=self.lr)
        else:
            self.opt = None  # PGD will step manually

        # Scheduler (optional)
        self.lr_scheduler = config.optimizer.exponentiallr
        self.gamma = config.optimizer.exponentiallr_gamma
        self.scheduler = ExponentialLR(self.opt, gamma=self.gamma) if (self.lr_scheduler and self.opt) else None

        # EOT / TV / Attn weights
        self.tv_weight = getattr(getattr(config, "loss", object()), "tv_weight", 1e-4)
        self.attn_w    = getattr(getattr(config, "loss", object()), "attn_hijack_w", 0.10)

        # PGD knobs (if enabled)
        self.pgd_steps = getattr(config.optimizer, "pgd_steps", 7)
        self.pgd_alpha = getattr(config.optimizer, "pgd_alpha", 2.0/255.0)

        # Metrics
        self.metric = SegmentationMetric(config)
        self.log_every = config.train.log_per_iters

        # Patch applier (uses your geometric constraints)
        self.apply_patch = Patch(config).apply_patch

    # ---------------------
    # Patch parametrization
    # ---------------------
    def get_patch(self):
        # R -> (0,1) with a margin
        return 0.5 * (torch.tanh(self.patch_param) + 1.0) * 0.999

    def init_lowfreq_tanh(self, shape, cutoff=0.2):
        C, H, W = shape
        device = self.device
        spec = torch.randn(C, H, W, dtype=torch.complex64, device=device)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        rad = (xx**2 + yy**2).sqrt()
        mask = (rad <= cutoff)
        spec = spec * mask
        img = torch.fft.ifft2(spec).real
        img = (img - img.amin(dim=(-2, -1), keepdim=True))
        img = img / (img.amax(dim=(-2, -1), keepdim=True) - img.amin(dim=(-2, -1), keepdim=True) + 1e-8)
        z = (img * 2 - 1).clamp(-0.999, 0.999)
        param = torch.atanh(z).detach()
        param.requires_grad_(True)
        return param

    # ---------------------
    # EOT on patch (keeps SxS)
    # ---------------------
    def eot_patch(self, patch_3chw):
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.85, 1.15)
        shear = [random.uniform(-5, 5), random.uniform(-5, 5)]
        out = TF.affine(patch_3chw, angle=angle, translate=[0, 0], scale=scale, shear=shear)
        out = TF.adjust_brightness(out, random.uniform(0.85, 1.15))
        out = TF.adjust_contrast(out,  random.uniform(0.85, 1.15))
        return out.clamp(0, 1)

    # ---------------------
    # TV regularizer
    # ---------------------
    def tv_loss(self, x):
        if x.dim() == 3:  # (C,H,W)
            tv_h = (x_
