import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.utilities.distributed import rank_zero_only
import matplotlib.pyplot as plt
import os
from pytorch_lightning.utilities import rank_zero_only

import csv
import os



class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")



# class LossLogger(Callback):
#     def __init__(self, log_dir='/scratch/YOURNAME/project/ControlNet/lightning_logs/loss_curves/'):
#         self.log_dir = log_dir
#         os.makedirs(self.log_dir, exist_ok=True)

#         # Explicitly store lists for specific metrics
#         self.loss_simple_step = []
#         self.loss_vlb_step = []
#         self.loss_step = []

#         # Path to the CSV file for saving loss values
#         self.csv_file_path = os.path.join(self.log_dir, 'loss_values.csv')
#         if not os.path.exists(self.csv_file_path):
#             # Create CSV file and write the header if it doesn't exist
#             with open(self.csv_file_path, mode='w', newline='') as csv_file:
#                 writer = csv.writer(csv_file)
#                 writer.writerow(['Epoch', 'loss_simple_step', 'loss_vlb_step', 'loss_step'])
            

#     @rank_zero_only
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         # Explicitly access and append specific metrics
#         logged_metrics = trainer.logged_metrics

#         if 'train/loss_simple_step' in logged_metrics:
#             self.loss_simple_step.append(logged_metrics['train/loss_simple_step'].item())

#         if 'train/loss_vlb_step' in logged_metrics:
#             self.loss_vlb_step.append(logged_metrics['train/loss_vlb_step'].item())

#         if 'train/loss_step' in logged_metrics:
#             self.loss_step.append(logged_metrics['train/loss_step'].item())

#     @rank_zero_only
#     def on_train_epoch_end(self, trainer, pl_module):
#         # Save the loss values to the CSV file
#         epoch = trainer.current_epoch + 1
#         avg_loss_simple = sum(self.loss_simple_step) / len(self.loss_simple_step) if self.loss_simple_step else 0
#         avg_loss_vlb = sum(self.loss_vlb_step) / len(self.loss_vlb_step) if self.loss_vlb_step else 0
#         avg_loss = sum(self.loss_step) / len(self.loss_step) if self.loss_step else 0

#         with open(self.csv_file_path, mode='a', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow([epoch, avg_loss_simple, avg_loss_vlb, avg_loss])

#         print(f"Saved loss values for epoch {epoch} to {self.csv_file_path}")

#         # Plot and save the loss curves for each explicitly recorded metric at the end of every epoch
#         metrics = {
#             'loss_simple_step': self.loss_simple_step,
#             'loss_vlb_step': self.loss_vlb_step,
#             'loss_step': self.loss_step,
#         }

#         for key, values in metrics.items():
#             plt.figure(figsize=(10, 5))
#             plt.plot(values, label=key)
#             plt.xlabel('Batch')
#             plt.ylabel('Loss')
#             plt.title(f'{key} Curve')
#             plt.legend()
#             plt.grid(True)

#             # Overwrite the plot for each epoch
#             curve_plot_path = os.path.join(self.log_dir, f'{key}_curve.png')
#             plt.savefig(curve_plot_path)
#             print(f"Updated loss curve for {key} at {curve_plot_path}")
#             plt.close()



class LossLogger(Callback):
    def __init__(self, log_dir='/scratch/YOURNAME/project/ControlNet/lightning_logs/loss_curves/'):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Explicitly store lists for specific metrics
        self.loss_simple_step = []
        self.loss_vlb_step = []
        self.loss_step = []

        # Path to the CSV file for saving loss values
        self.csv_file_path = os.path.join(self.log_dir, 'loss_values.csv')
        if not os.path.exists(self.csv_file_path):
            # Create CSV file and write the header if it doesn't exist
            with open(self.csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Epoch', 'loss_simple_step', 'loss_vlb_step', 'loss_step'])
            

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Explicitly access and append specific metrics
        logged_metrics = trainer.logged_metrics

        if 'train/loss_simple_step' in logged_metrics:
            self.loss_simple_step.append(logged_metrics['train/loss_simple_step'].item())

        if 'train/loss_vlb_step' in logged_metrics:
            self.loss_vlb_step.append(logged_metrics['train/loss_vlb_step'].item())

        if 'train/loss_step' in logged_metrics:
            self.loss_step.append(logged_metrics['train/loss_step'].item())

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Save the loss values to the CSV file
        epoch = trainer.current_epoch + 1
        avg_loss_simple = sum(self.loss_simple_step) / len(self.loss_simple_step) if self.loss_simple_step else 0
        avg_loss_vlb = sum(self.loss_vlb_step) / len(self.loss_vlb_step) if self.loss_vlb_step else 0
        avg_loss = sum(self.loss_step) / len(self.loss_step) if self.loss_step else 0

        with open(self.csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch, avg_loss_simple, avg_loss_vlb, avg_loss])

        print(f"Saved loss values for epoch {epoch} to {self.csv_file_path}")

        # Plot and save the loss curves for each explicitly recorded metric at the end of every epoch
        metrics = {
            'loss_simple_step': self.loss_simple_step,
            'loss_vlb_step': self.loss_vlb_step,
            'loss_step': self.loss_step,
        }

        for key, values in metrics.items():
            plt.figure(figsize=(10, 5))
            plt.plot(values, label=key)
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title(f'{key} Curve')
            plt.legend()
            plt.grid(True)

            # Overwrite the plot for each epoch
            curve_plot_path = os.path.join(self.log_dir, f'{key}_curve.png')
            plt.savefig(curve_plot_path)
            print(f"Updated loss curve for {key} at {curve_plot_path}")
            plt.close()
