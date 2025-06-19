import torch
import torch.nn.functional as F
import torch.nn as nn
from patch.create import Patch

class LinearScheduler:
    """
    Simple linear scheduler for a value from start to end over total_epochs.
    """
    def __init__(self, start_value, end_value, total_epochs):
        self.start = start_value
        self.end = end_value
        self.total = max(total_epochs, 1)

    def get(self, epoch):
        e = min(max(epoch, 0), self.total)
        return self.start + (self.end - self.start) * (e / self.total)

class PatchLoss(nn.Module):
    def __init__(self, config, feature_extractor=None):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label
        self.feature_extractor = feature_extractor

        # schedulers
        E1 = config.attack.stage1_epochs
        E2 = config.attack.stage2_epochs
        self.gamma_sched = LinearScheduler(config.attack.gamma_start,
                                          config.attack.gamma_end, E1)
        self.beta_sched  = LinearScheduler(config.attack.beta_start,
                                          config.attack.beta_end,  E2)
        self.current_epoch = 0
        self.register_buffer('ema_kl', torch.zeros(1))

        # hyper-params
        self.margin = getattr(config.attack, 'margin', 0.1)
        self.lambda_ent = getattr(config.attack, 'lambda_ent', 0.1)
        self.eta = getattr(config.attack, 'eta', 0.5)
        self.use_feat_div = getattr(config.attack, 'use_feat_div', False)

    def step_epoch(self):
        self.current_epoch += 1

    @property
    def gamma(self):
        return self.gamma_sched.get(self.current_epoch)

    @property
    def beta(self):
        e2 = max(self.current_epoch - self.config.attack.stage1_epochs, 0)
        return self.beta_sched.get(e2)

    def _make_beta_map(self, kl_map, target):
        mask = (target != self.ignore_label)
        # EMA update
        ema_new = 0.9 * self.ema_kl + 0.1 * kl_map.detach().mean()
        self.ema_kl = ema_new

        # per-sample var
        var = ((kl_map - ema_new)**2 * mask).sum(dim=[1,2])
        var /= (mask.sum(dim=[1,2]) + 1e-8)

        # normalize
        vmin, vmax = var.min(), var.max()
        norm = (var - vmin) / (vmax - vmin + 1e-8)
        beta_map = norm.view(-1,1,1).expand_as(kl_map)
        return beta_map


    def _make_margin_loss(self, pred, target):
        N, C, H, W = pred.shape
    
        # Ensure correct shape
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
    
        # Ensure dtype and device match
        target = target.to(dtype=torch.long, device=pred.device)
    
        assert target.shape == (N, H, W), f"Expected target shape [N,H,W], got {target.shape}"
        
        logits = pred  # [N,C,H,W]
        true_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)  # [N,H,W]
    
        inf_mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), float('-inf'))
        wrong_logit, _ = (logits + inf_mask).max(dim=1)
        margin_loss = F.relu(true_logit - wrong_logit + self.margin)
        return margin_loss


    def compute_loss_adaptive(self, pred, target, clean_pred=None, clean_image=None):
        N, C, H, W = pred.shape
        ignore = (target == self.ignore_label)

        # masks
        pred_labels = pred.argmax(dim=1)
        correct = (pred_labels == target) & ~ignore
        broken  = (pred_labels != target) & ~ignore

        # choose per-pixel loss: margin or CE
        if self.config.attack.use_margin:
            base_loss = self._make_margin_loss(pred, target)
        else:
            base_loss = F.cross_entropy(pred, target,
                                        ignore_index=self.ignore_label,
                                        reduction='none')

        # Stage 1 term
        γ = self.gamma
        L1 = ((1 - γ) * base_loss * correct).sum()

        # Stage 2 term
        # KL map
        pred_log   = F.log_softmax(pred, dim=1)
        clean_soft = F.softmax(clean_pred, dim=1)
        kl_map = F.kl_div(pred_log, clean_soft, reduction='none').sum(1)
        # beta map
        β_map = self._make_beta_map(kl_map, target)
        # high/low masks
        thresh = kl_map[~ignore].mean()
        high = (kl_map > thresh) & ~ignore
        low  = (kl_map <= thresh) & ~ignore

        # CE variant of Stage2
        L2_ce = ((1 - β_map) * base_loss * high + β_map * base_loss * low).sum()

        # optional feature-divergence
        if self.use_feat_div and self.feature_extractor is not None and clean_image is not None:
            F_adv   = self.feature_extractor(pred)
            F_clean = self.feature_extractor(clean_image)
            # cosine dist
            num = (F_adv * F_clean).sum(dim=1)
            den = F_adv.norm(dim=1) * F_clean.norm(dim=1) + 1e-8
            feat_div = 1 - num/den  # [N,h',w']
            up = F.interpolate(feat_div.unsqueeze(1), (H,W), mode='bilinear').squeeze(1)
            mask = ~ignore
            mth = up[mask].mean()
            high_f = mask & (up > mth)
            low_f  = mask & (up <= mth)
            L2_feat = ((1 - self.beta) * base_loss * high_f +
                       self.beta * base_loss * low_f).sum()
        else:
            L2_feat = 0.0

        # entropy regularization
        p_adv = F.softmax(pred, dim=1)
        ent = -(p_adv * p_adv.log()).sum(dim=1)
        ent_loss = (ent * (~ignore)).sum() / ((~ignore).sum() + 1e-8)

        # combine Stage2
        L2 = L2_ce + (L2_feat if L2_feat else 0)

        # final mix
        eta = self.eta
        L = (1 - eta) * L1 + eta * L2 + self.lambda_ent * ent_loss

        total = float((~ignore).sum().item() + 1e-8)
        return L / total


    def compute_loss(self, model_output, label):
        """
        Compute the adaptive loss function
        """

        #print(model_output.shape,true_labels.shape)
        #print(model_output.argmax(dim=1).shape)
        ce_loss = nn.CrossEntropyLoss(reduction="none",
                                      ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss_map = ce_loss(model_output, label.long())  # Compute loss for all pixels
        #print(f'loss map: {loss_map.shape}')
              
      
        # Get correctly classified and misclassified pixel sets
        predict = torch.argmax(model_output, 1).float() + 1
        target = label.float() + 1
        target[target>=255] = 0
        # print(predict.dtype,predict.shape,target.dtype,target.shape)
        # temp1 = (predict == target).float()
        # temp2 = (target>0).float()
        # print(temp1.dtype,temp2.dtype)
        correct_mask = (predict == target)*(target > 0)
        incorrect_mask = (predict != target)*(target > 0)  # Opposite of correctly classified
        #print(f'Correct mask: {correct_mask.shape}')  
        # Compute separate loss terms
        loss_correct = (loss_map * correct_mask).sum()/correct_mask.sum()  # Loss on correctly classified pixels
        loss_incorrect = (loss_map * incorrect_mask).sum()/incorrect_mask.sum()  # Loss on already misclassified pixels

        # Compute adaptive balancing factor
        num_correct = correct_mask.sum()
        num_total = (target != 0).sum()
        gamma = num_correct / num_total  # Avoid division by zero

        # Final adaptive loss
        loss = gamma * loss_correct + (1 - gamma) * loss_incorrect
        # print(f'Gamma:{gamma}')
        # print(f'loss correct:{loss_correct}')
        # print(f'loss incorrect: {loss_incorrect}')
        #return loss
        return loss_correct 


    def compute_loss_direct(self, model_output, label):
        """
        Compute the adaptive loss function
        """
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss = ce_loss(model_output, label.long())  # Compute loss for all pixels
        return loss 
