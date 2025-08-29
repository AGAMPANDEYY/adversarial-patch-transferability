import sys, time, datetime, random
from typing import List, Tuple


original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from dataset.cityscapes import Cityscapes
from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss          
from patch.create import Patch               
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
            output_attentions=True,            # attention hijack
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

        # Patch applier
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
            tv_h = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
            tv_w = (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
        elif x.dim() == 4:
            tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
            tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        else:
            raise ValueError("tv_loss expects 3D or 4D tensor")
        return tv_h + tv_w

    # ---------------------
    # Attention Hijack loss
    # ---------------------
    @torch.no_grad()
    def _estimate_patch_mask(self, clean_img, patched_img, thresh=1e-3):
        # Derive a binary mask of where the patch was pasted (B,1,H,W)
        diff = (patched_img - clean_img).abs().sum(dim=1, keepdim=True)
        return (diff > thresh).float()

    def attn_hijack_loss(self, attentions: Tuple[torch.Tensor, ...], patch_mask: torch.Tensor,
                         H: int, W: int) -> torch.Tensor:
        """
        attentions: tuple of attention tensors from SegFormer encoder blocks.
                    Each att has shape (B, heads, N, N).
        patch_mask: (B,1,H,W) in {0,1}
        We project patch_mask to token grids by trying strides {4,8,16,32} and
        picking the one matching N = (H/stride)*(W/stride).
        We maximize mean attention mass TO patch tokens (columns).
        """
        if attentions is None or len(attentions) == 0:
            return torch.zeros((), device=self.device)

        strides = [4, 8, 16, 32]
        B = patch_mask.size(0)
        hijack_loss = 0.0
        blocks_count = 0

        for att in attentions:
            # att: (B, heads, N, N)
            if att is None:
                continue
            B_a, Hh, Nq, Nk = att.shape
            assert B_a == B and Nq == Nk, "Unexpected attention shape"

            # find which stride matches this N
            stride = None
            for s in strides:
                if (H // s) * (W // s) == Nq:
                    stride = s
                    break
            if stride is None:
                continue

            # downsample patch mask to this token grid
            h_s, w_s = H // stride, W // stride
            pmask = F.interpolate(patch_mask, size=(h_s, w_s), mode="nearest")  # (B,1,h_s,w_s)
            pmask = pmask.view(B, -1)  # (B, N)

            # average over heads -> (B, N, N)
            att_mean = att.mean(dim=1)

            # For each batch item: mean over queries of attention TO patch tokens (columns where pmask==1)
            batch_loss = 0.0
            valid = 0
            for b in range(B):
                idx = pmask[b] > 0.5
                if idx.sum() == 0:
                    continue
                # att_mean[b]: (N, N) => [:, idx] selects attention to patch tokens
                # We want to maximize this mass => minimize negative mean
                mass_to_patch = att_mean[b][:, idx].mean()
                batch_loss += (-mass_to_patch)
                valid += 1
            if valid > 0:
                hijack_loss = hijack_loss + (batch_loss / valid)
                blocks_count += 1

        if blocks_count == 0:
            return torch.zeros((), device=self.device)
        return hijack_loss / blocks_count

    # ---------------------
    # Forward helper (SegFormer)
    # ---------------------
    def segformer_forward(self, img_4bhwc):
        """
        img: (B,3,H,W) in the normalization used by your dataset (works fine).
        Returns logits (B,C,H,W) resized by the HF model, and attentions tuple.
        """
        out = self.model(img_4bhwc, output_attentions=True)
        # HF returns dict-like: out.logits (B,num_labels,H,W), out.attentions (tuple)
        return out.logits, out.attentions

    # ---------------------
    # Train
    # ---------------------
    def train(self):
        start_epoch, end_epoch, total_epochs = self.start_epoch, self.end_epoch, self.total_epochs
        assert total_epochs == 30, f"This schedule expects 30 epochs; got {total_epochs}."
        switch_epoch = start_epoch + (total_epochs // 2)  # Stage-1 then Stage-2(JS)

        start_time = time.time()
        self.log.info(
            f"Start training | Total Epochs: {total_epochs} "
            f"(Stage-1: {start_epoch}–{switch_epoch-1}, Stage-2(JS): {switch_epoch}–{end_epoch-1}) | "
            f"Iterations/epoch: {self.iters_per_epoch}"
        )

        IoU_over_epochs = []
        H, W = self.cfg.train.height, self.cfg.train.width

        for ep in range(start_epoch, end_epoch):
            self.metric.reset()
            use_stage1 = (ep < switch_epoch)
            stage = "Stage-1" if use_stage1 else "Stage-2(JS)"
            self.log.info(f"Epoch {ep}: using {stage}")

            cum_attack_loss = 0.0
            for it, batch in enumerate(self.train_dl, 0):
                image, true_label, _, _, _ = batch
                image = image.to(self.device)             # (B,3,H,W)
                true_label = true_label.to(self.device)   # (B,H,W)

                # Build current visible patch and EOT-transform it
                base_patch = self.get_patch()             # (3,S,S)
                patch = self.eot_patch(base_patch)        # (3,S,S)

                # Paste the patch
                patched_image, patched_label = self.apply_patch(image, true_label, patch)
                patched_label = patched_label.long()

                # Forward passes (ViT target)
                logits_adv, atts_adv = self.segformer_forward(patched_image)
                with torch.no_grad():
                    logits_clean, _ = self.segformer_forward(image)

                # Your 2-stage attack loss
                if use_stage1:
                    attack_loss = self.criterion.compute_loss_transegpgd_stage1(
                        logits_adv, patched_label, logits_clean
                    )
                else:
                    attack_loss = self.criterion.compute_loss_transegpgd_stage2_js(
                        logits_adv, patched_label, logits_clean
                    )

                # Attention Hijack loss (maximize attention mass to patch)
                with torch.no_grad():
                    patch_mask = self._estimate_patch_mask(image, patched_image)  # (B,1,H,W)
                ah_loss = self.attn_hijack_loss(atts_adv, patch_mask, H, W)

                # TV on base patch
                tv = self.tv_loss(base_patch)

                # Total objective (gradient ASCENT on attack; so minimize negative)
                total = (-attack_loss) + (self.tv_weight * tv) + (self.attn_w * ah_loss)

                # === Optimize the patch ===
                # Model is frozen; we only backprop into patch_param
                if not self.use_pgd:
                    # Adam step on patch_param
                    # zero grads on "model" just in case
                    self.model.zero_grad(set_to_none=True)
                    if self.patch_param.grad is not None:
                        self.patch_param.grad.zero_()
                    total.backward()
                    # step
                    self.opt.step()
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    # PGD inner loop (K small steps)
                    K = self.pgd_steps
                    alpha = self.pgd_alpha
                    for _ in range(K):
                        base_patch = self.get_patch()
                        patch = self.eot_patch(base_patch)

                        patched_image, patched_label = self.apply_patch(image, true_label, patch)
                        patched_label = patched_label.long()

                        logits_adv, atts_adv = self.segformer_forward(patched_image)
                        with torch.no_grad():
                            logits_clean, _ = self.segformer_forward(image)

                        if use_stage1:
                            attack_loss = self.criterion.compute_loss_transegpgd_stage1(
                                logits_adv, patched_label, logits_clean
                            )
                        else:
                            attack_loss = self.criterion.compute_loss_transegpgd_stage2_js(
                                logits_adv, patched_label, logits_clean
                            )
                        with torch.no_grad():
                            patch_mask = self._estimate_patch_mask(image, patched_image)
                        ah_loss = self.attn_hijack_loss(atts_adv, patch_mask, H, W)
                        tv = self.tv_loss(base_patch)

                        total_inner = (-attack_loss) + (self.tv_weight * tv) + (self.attn_w * ah_loss)

                        self.model.zero_grad(set_to_none=True)
                        if self.patch_param.grad is not None:
                            self.patch_param.grad.zero_()
                        total_inner.backward()

                        with torch.no_grad():
                            self.patch_param += alpha * self.patch_param.grad.sign()
                            # Re-project to valid tanh-param range via visible clamp
                            vis = self.get_patch().clamp(0, 1)
                            z = (vis * 2 - 1).clamp(-0.999, 0.999)
                            self.patch_param.copy_(torch.atanh(z))

                # Metrics on the attacked output
                self.metric.update(logits_adv, patched_label)
                _, mIoU = self.metric.get()

                # Logging
                cum_attack_loss += attack_loss.item()
                if it % self.log_every == 0:
                    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                    eta_sec = ((time.time() - start_time) / max(1, (ep - self.start_epoch) * self.iters_per_epoch + it + 1)) * \
                              (self.total_epochs * self.iters_per_epoch - ((ep - self.start_epoch) * self.iters_per_epoch + it + 1))
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                    lr_show = (self.opt.param_groups[0]["lr"] if self.opt is not None else 0.0)
                    self.log.info(
                        f"Epoch {ep}/{end_epoch} || Batch {it+1}/{self.iters_per_epoch} || "
                        f"LR: {lr_show:.3e} || AttackLoss: {attack_loss.item():.4f} || "
                        f"AttnHijack: {ah_loss.item():.4f} || TV: {tv.item():.4f} || mIoU: {mIoU:.4f} || "
                        f"Elapsed: {elapsed} || ETA: {eta_str}"
                    )

            # epoch summary
            pixAcc, meanIoU = self.metric.get()
            avg_attack = cum_attack_loss / max(1, self.iters_per_epoch)
            self.log.info("-" * 100)
            self.log.info(
                f"Epoch {ep}/{end_epoch} | {stage} | Avg AttackLoss: {avg_attack:.4f} | "
                f"Avg mIoU: {meanIoU:.4f} | Avg pixAcc: {pixAcc:.4f}"
            )
            self.log.info("-" * 100)
            IoU_over_epochs.append(self.metric.get(full=True))

        # Return the visible patch and IoU logs
        return self.get_patch().detach(), np.array(IoU_over_epochs)
