import sys
# Save the original sys.path
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")
from dataset.cityscapes import Cityscapes

from pretrained_models.models import Models

from pretrained_models.ICNet.icnet import ICNet
from pretrained_models.BisNetV1.model import BiSeNetV1
from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model

from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch
from torch.optim.lr_scheduler import ExponentialLR
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF

# Restore original sys.path to avoid conflicts or shadowing
sys.path = original_sys_path

class PatchTrainer():
  def __init__(self,config,main_logger):
      self.config = config
      self.start_epoch = config.train.start_epoch
      self.end_epoch = config.train.end_epoch
      self.epochs = self.end_epoch - self.start_epoch
      self.batch_train = config.train.batch_size
      self.batch_test = config.test.batch_size
      self.device = config.experiment.device
      self.logger = main_logger
      self.lr = config.optimizer.init_lr
      self.power = config.train.power
      self.lr_scheduler = config.optimizer.exponentiallr
      self.lr_scheduler_gamma = config.optimizer.exponentiallr_gamma
      self.log_per_iters = config.train.log_per_iters
      self.patch_size = config.patch.size
      self.apply_patch = Patch(config).apply_patch
      self.epsilon = config.optimizer.init_lr

      cityscape_train = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.train,
          num_classes = config.dataset.num_classes,
          multi_scale = config.train.multi_scale,
          flip = config.train.flip,
          ignore_label = config.train.ignore_label,
          base_size = config.train.base_size,
          crop_size = (config.train.height,config.train.width),
          scale_factor = config.train.scale_factor
        )

      cityscape_test = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.val,
          num_classes = config.dataset.num_classes,
          multi_scale = False,
          flip = False,
          ignore_label = config.train.ignore_label,
          base_size = config.test.base_size,
          crop_size = (config.test.height,config.test.width),
        )
      
      self.train_dataloader = torch.utils.data.DataLoader(dataset=cityscape_train,
                                              batch_size=self.batch_train,
                                              shuffle=config.train.shuffle,
                                              num_workers=config.train.num_workers,
                                              pin_memory=config.train.pin_memory,
                                              drop_last=config.train.drop_last)
      self.test_dataloader = torch.utils.data.DataLoader(dataset=cityscape_test,
                                            batch_size=self.batch_test,
                                            shuffle=False,
                                            num_workers=config.test.num_workers,
                                            pin_memory=config.test.pin_memory,
                                            drop_last=config.test.drop_last)
      


      self.iters_per_epoch = len(self.train_dataloader)
      self.max_iters = self.end_epoch * self.iters_per_epoch

      ## Getting the model
      self.model = Models(self.config)
      self.model.get()

      ## loss
      self.criterion = PatchLoss(self.config)

      ## optimizer
      # Initialize adversarial patch (random noise)
      self.patch = torch.rand((3, self.patch_size, self.patch_size), 
                              requires_grad=True, 
                              device=self.device)

      # ===== NEW: tanh-parameterized patch with low-frequency init =====
      self.patch_param = self.init_lowfreq_tanh((3, self.patch_size, self.patch_size), cutoff=0.2)

      # (optional) choose another init by swapping the line above with:
      # self.patch_param = self.init_perlin_tanh((3, self.S, self.S))
      # self.patch_param = self.init_dataset_color_tanh((3, self.S, self.S), mean=(0.286,0.325,0.283))

      # Optimizer (Adam is typically more stable than per-step FGSM)
      self.opt = torch.optim.Adam([self.patch_param], lr=self.lr)
      self.scheduler = ExponentialLR(self.opt, gamma=self.lr_scheduler_gamma) if self.lr_scheduler else None

      # TV regularizer weight (can expose to config)
      self.tv_weight = getattr(config.loss, 'tv_weight', 1e-4)

      
      # # Define optimizer
      # self.optimizer = torch.optim.SGD(params = [self.patch],
      #                         lr=self.lr,
      #                         momentum=config.optimizer.momentum,
      #                         weight_decay=config.optimizer.weight_decay,
      #                         nesterov=config.optimizer.nesterov,
      # )
      # if self.lr_scheduler:
      #   self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)


      ## Initializing quantities
      self.metric = SegmentationMetric(config) 
      self.current_mIoU = 0.0
      self.best_mIoU = 0.0

      self.current_epoch = 0
      self.current_iteration = 0

  # ---------------------
  # Patch parametrization & inits
  # ---------------------
  def get_patch(self):
      # maps R -> [0,1] smoothly and avoids exact 0/1 saturation
      return 0.5 * (torch.tanh(self.patch_param) + 1.0) * 0.999

  def init_lowfreq_tanh(self, shape, cutoff=0.2):
      """Low-frequency FFT init in [0,1], then inverse tanh to get patch_param."""
      C,H,W = shape
      device = self.device
      # random complex spectrum
      spec = torch.randn(C,H,W, dtype=torch.complex64, device=device)
      yy, xx = torch.meshgrid(
          torch.linspace(-1,1,H,device=device),
          torch.linspace(-1,1,W,device=device), indexing='ij')
      rad = (xx**2 + yy**2).sqrt()
      mask = (rad <= cutoff)
      spec = spec * mask  # keep low-freq only
      img = torch.fft.ifft2(spec).real
      # normalize to [0,1]
      img = (img - img.amin(dim=(-2,-1), keepdim=True))
      img = img / (img.amax(dim=(-2,-1), keepdim=True) - img.amin(dim=(-2,-1), keepdim=True) + 1e-8)
      # inverse tanh
      z = (img*2 - 1).clamp(-0.999, 0.999)
      param = torch.atanh(z).detach().to(device)
      param.requires_grad_(True)
      return param

  def init_perlin_tanh(self, shape):
      import torch.nn.functional as F
      C,H,W = shape; device = self.device
      def perlin_octave(freq, amp):
          grid = torch.rand(2, freq+1, freq+1, device=device)
          noise = F.interpolate(grid.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True)[0]
          # simple directional mix
          xs = torch.linspace(0,1,W,device=device)
          ys = torch.linspace(0,1,H,device=device)
          n = noise[0][None,:,:]*xs + noise[1][:,None]*ys
          return amp * (n - n.min()) / (n.max()-n.min()+1e-8)
      base = sum(perlin_octave(f,a) for f,a in [(4,0.5),(8,0.25),(16,0.15),(32,0.10)])
      base = base.clamp(0,1)
      img = torch.stack([base for _ in range(C)], dim=0)
      z = (img*2 - 1).clamp(-0.999, 0.999)
      param = torch.atanh(z).detach(); param.requires_grad_(True)
      return param.to(device)

  def init_dataset_color_tanh(self, shape, mean=(0.286,0.325,0.283)):
      C,H,W = shape; device = self.device
      mean = torch.tensor(mean, device=device)[:,None,None]
      lowf = (torch.rand(C,H,W, device=device)*0.1 - 0.05)
      img = (mean + lowf).clamp(0,1)
      z = (img*2 - 1).clamp(-0.999, 0.999)
      param = torch.atanh(z).detach(); param.requires_grad_(True)
      return param.to(device)

  # ---------------------
  # Light patch-space EOT (keeps patch size SxS)
  # ---------------------
  def eot_transform_patch(self, patch):
      # patch: (3,S,S) in [0,1]
      angle = random.uniform(-20, 20)
      scale = random.uniform(0.8, 1.2)
      shear = [random.uniform(-5,5), random.uniform(-5,5)]
      # affine keeps size; translate kept 0 because apply_patch chooses location
      patch_t = TF.affine(patch, angle=angle, translate=[0,0], scale=scale, shear=shear)
      # mild color jitter to simulate print/camera shifts
      patch_t = TF.adjust_brightness(patch_t, random.uniform(0.85, 1.15))
      patch_t = TF.adjust_contrast(patch_t,  random.uniform(0.85, 1.15))
      return patch_t.clamp(0,1)

  # ---------------------
  # Total variation for smoothness / physical robustness
  # ---------------------
  def tv_loss(self, x):
      return ((x[:,:,:-1,:]-x[:,:,1:,:]).abs().mean() +
              (x[:,:,:,:-1]-x[:,:,:,1:]).abs().mean())

  def train(self):
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
    start_epoch=0
    switch_epoch=(start_epoch+self.end_epoch)//2

    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, iters_per_epoch))
    IoU = []
    for ep in range(self.start_epoch, self.end_epoch):
      self.current_epoch = ep
      self.metric.reset()
      total_loss = 0
      samplecnt = 0
      for i_iter, batch in enumerate(self.train_dataloader, 0):
          self.current_iteration += 1
          samplecnt += batch[0].shape[0]
          image, true_label,_, _, _ = batch
          image, true_label = image.to(self.device), true_label.to(self.device)
          samplecnt += image.shape[0]

          # ---- get current patch & apply light EOT on patch ----
          patch = self.get_patch()                     # (3,S,S) in [0,1]
          patch = self.eot_transform_patch(patch)      # optional EOT on patch itself

          # ---- paste patch (your existing function) ----
          patched_image, patched_label = self.apply_patch(image, true_label, patch)      
          
          # Randomly place patch in image and label(put ignore index)
          #patched_image, patched_label = self.apply_patch(image,true_label,self.patch)
          # fig = plt.figure()
          # ax = fig.add_subplot(1,2,1)
          # ax.imshow(patched_image[0].permute(1,2,0).cpu().detach().numpy())
          # ax = fig.add_subplot(1,2,2)
          # ax.imshow(patched_label[0].cpu().detach().numpy())
          # plt.show()

          # Forward pass through the model (and interpolation if needed)
          output = self.model.predict(patched_image,patched_label.shape)
          #plt.imshow(output.argmax(dim =1)[0].cpu().detach().numpy())
          #plt.show()
          #break
          with torch.no_grad():
             clean_output = self.model.predict(image, patched_label.shape)

          with torch.no_grad():
              pred_labels = output.argmax(dim=1)  # (N, H, W)
              correct_pixels = (pred_labels == patched_label) & (patched_label != self.config.train.ignore_label)
              num_correct = correct_pixels.sum().item()
          
          if num_correct > 0:
              self.logger.info(f"Batch {i_iter}: {num_correct} correctly predicted pixels remaining.")
              loss = self.criterion.compute_loss_transegpgd_stage1(output, patched_label, clean_output)
          else:
              loss = self.criterion.compute_loss_transegpgd_stage2(output, patched_label, clean_output)
              # Compute adaptive loss
              #loss = self.criterion.compute_loss(output, patched_label)
              #loss = self.criterion.compute_loss_direct(output, patched_label)

          #loss = self.criterion.compute_loss_transegpgd(output, patched_label, clean_output)

          total_loss += loss.item()
          #break

          ## metrics
          self.metric.update(output, patched_label)
          pixAcc, mIoU = self.metric.get()

          # Backpropagation
          self.model.model.zero_grad()
          if self.patch.grad is not None:
            self.patch.grad.zero_()
          loss.backward()
          with torch.no_grad():
              #self.patch += self.epsilon * self.patch.grad.sign()  # Update patch using FGSM-style ascent
              self.patch += self.epsilon * self.patch.grad.data.sign()
              self.patch.clamp_(0, 1)  # Keep pixel values in valid range

          ## ETA
          eta_seconds = ((time.time() - start_time) / self.current_iteration) * (iters_per_epoch*epochs - self.current_iteration)
          eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

          if i_iter % self.log_per_iters == 0:
            self.logger.info(
              "Epochs: {:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                  self.current_epoch, self.end_epoch,
                  samplecnt, self.batch_train*iters_per_epoch,
                  #self.optimizer.param_groups[0]['lr'],
                  self.epsilon,
                  loss.item(),
                  mIoU,
                  str(datetime.timedelta(seconds=int(time.time() - start_time))),
                  eta_string))
          

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = total_loss/len(self.train_dataloader)
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))

      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

      IoU.append(self.metric.get(full=True))

    return self.patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs

    
