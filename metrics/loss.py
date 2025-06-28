import torch
import torch.nn.functional as F
import torch.nn as nn
from patch.create import Patch

class PatchLoss(nn.Module):
    def __init__(self, config):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label
        self.apply_patch = Patch(config).apply_patch
        self.ignore_index= config.train.ignore_label
        self.feature_extractor = feature_extractor

        # schedulers
        E1 = config.attack.stage1_epochs
        E2 = config.attack.stage2_epochs
        self.gamma_sched = LinearScheduler(config.attack.gamma_start,
                                          config.attack.gamma_end, E1)
        self.beta_sched  = LinearScheduler(config.attack.beta_start,
                                          config.attack.beta_end,  E2)
        self.current_epoch = 0
        self.register_buffer('ema_kl', torch.zeros(1, device=self.device))

        # hyper-params
        self.margin = getattr(config.attack, 'margin', 0.1)
        self.lambda_ent = getattr(config.attack, 'lambda_ent', 0.1)
        self.eta = getattr(config.attack, 'eta', 0.5)
        self.use_feat_div = getattr(config.attack, 'use_feat_div', False)

    def compute_loss_transegpgd_stage1(self, pred, target, clean_pred):
        """
        Stage 1: emphasize hard-to-attack pixels (correctly predicted ones).
        """
        N, C, H, W = pred.shape
        pred_softmax = F.softmax(pred, dim=1)
        target_flat = target.view(-1)
        pred_label = pred_softmax.argmax(dim=1)

        # Flatten for per-pixel comparison
        pred_label_flat = pred_label.view(-1)
        correct_mask = (pred_label_flat == target_flat) & (target_flat != self.ignore_index)
        incorrect_mask = (pred_label_flat != target_flat) & (target_flat != self.ignore_index)

        loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none').view(-1)

        total_pixels = float(correct_mask.sum() + incorrect_mask.sum() + 1e-8)

        loss_weighted = (1 - self.gamma) * loss[correct_mask].sum() + \
                        self.gamma * loss[incorrect_mask].sum()

        return loss_weighted / total_pixels

    def compute_loss_transegpgd_stage2(self, pred, target, clean_pred):
        """
        Stage 2: emphasize high-transferability pixels (large KL divergence from clean prediction).
        """
        pred_softmax = F.softmax(pred, dim=1)
        clean_softmax = F.softmax(clean_pred, dim=1)

        kl_div = F.kl_div(pred_softmax.log(), clean_softmax, reduction='none').sum(1)  # (N, H, W)
        kl_mean = kl_div[target != self.ignore_index].mean()

        high_transfer_mask = (kl_div > kl_mean) & (target != self.ignore_index)
        low_transfer_mask = (kl_div <= kl_mean) & (target != self.ignore_index)

        loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')

        total_pixels = float(high_transfer_mask.sum() + low_transfer_mask.sum() + 1e-8)

        loss_weighted = (1 - self.beta) * loss[high_transfer_mask].sum() + \
                        self.beta * loss[low_transfer_mask].sum()

        return loss_weighted / total_pixels

    def compute_loss_transegpgd(self,output, patched_label, clean_output):

        # 1) per-pixel CE over *patched* region
        ce = F.cross_entropy(output, patched_label, reduction='none')   # [B,H,W]
        patch_mask = (self.apply_patch(torch.zeros_like(image), 
                      torch.zeros_like(patched_label), self.patch)[0] 
                      > 0).float()  # [H,W], 1 inside patch region
        ce_patch = ce * patch_mask  # zero-out non-patch pixels
        
        # Stage 1: hard-pixel focus
        preds = output.argmax(dim=1)                                 # [B,H,W]
        correct = (preds == patched_label).float() * patch_mask     # Ω_F
        wrong   = (1 - correct) * patch_mask                        # Ω_T
        γ = self.config.attack.gamma  # e.g. 0.7
        L1 = ((1-γ)*(ce_patch * wrong).sum() 
              + γ*(ce_patch * correct).sum()) / patch_mask.sum()
        
        # Stage 2: KL‑seed boosting
        p_adv   = F.softmax(output, dim=1)
        p_clean = F.softmax(clean_output, dim=1)
        D_kl    = (p_adv * (p_adv.log() - p_clean.log())).sum(dim=1)  # [B,H,W]
        mean_kl = D_kl.mean(dim=[1,2], keepdim=True)
        high_kl = ((D_kl > mean_kl).float() * patch_mask)           # P_H
        low_kl  = ((D_kl <= mean_kl).float() * patch_mask)          # P_L
        β = self.config.attack.beta  # e.g. 0.2
        L2 = ((1-β)*(ce_patch * high_kl).sum() 
              + β*(ce_patch * low_kl).sum()) / patch_mask.sum()
        
        # Combine
        η = self.config.attack.eta  # e.g. 0.5
        loss = (1-η)*L1 + η*L2
        return loss


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
