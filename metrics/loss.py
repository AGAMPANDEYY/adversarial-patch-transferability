import torch
import torch.nn.functional as F
import torch.nn as nn

class PatchLoss(nn.Module):
    def __init__(self, config):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label

    def compute_loss_transegpgd(self,ouput, patched_label, clean_output):

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
