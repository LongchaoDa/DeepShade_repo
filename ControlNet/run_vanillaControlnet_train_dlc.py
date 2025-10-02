from share import *
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger, LossLogger
from cldm.model import create_model, load_state_dict
import random, os
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime


# dataset = ['Beijing', 'Phoenix', 'Sau Paulo', 'Madrid', 'Cairo', 'Mumbai', 'Tempe']
# select_data = dataset[-1]



# Configs
resume_path = '/scratch/YOURNAME/project/plantShade/ControlNet/models/control_sd21_ini.ckpt' # initial checkpoint

# resume_path = '/scratch/YOURNAME/project/ControlNet/models/epoch_51_step_1351.ckpt'

batch_size = 16  # Adjust batch size as needed
logger_freq = 300
learning_rate = 1e-4
sd_locked = True
only_mid_control = False

# Define your dataset class (as before)
import json
import cv2
import numpy as np
from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, seed=42):
        self.data = []
        # v2: complete files
        with open(f'/scratch/YOURNAME/project/plantShade/dataset/Tempe/train_ok.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
                random.seed(seed)
                random.shuffle(self.data)# 1/ shuffle the data at the first time when it is loaded, make sure it is shuffled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

        # Do not forget that OpenCV reads images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    



if __name__ == '__main__':
    # First use CPU to load models. PyTorch Lightning will automatically move it to GPUs.
    model = create_model('/scratch/YOURNAME/project/ControlNet/models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Dataset and DataLoader
    dataset = MyDataset()
    print('There are data size of:', len(dataset))
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True) # 2/ set the shuffle is True, to make sure it is actually shuffled (double guarantee)

    logger = ImageLogger(batch_frequency=logger_freq)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # training_dir = f"/scratch/YOURNAME/project/ControlNet/lightning_logs/inDomain/{select_data}/" + time_str + "/"
    training_dir = f"/scratch/YOURNAME/project/plantShade/ControlNet/0out/ControlNet_vanilla_Tempe/" + time_str + "/"


    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    loss_logger = LossLogger(log_dir= training_dir+'loss_curves/')
    

    best_checkpoint_callback = ModelCheckpoint(
    dirpath= training_dir + 'best/',
    filename='best-{epoch:02d}-{train_loss_simple_step:.4f}',  # Save best checkpoint based on train/loss_simple_step
    monitor='train/loss_simple_step',                         # Monitor train/loss_simple_step
    mode='min',                                               # Save the lowest train/loss_simple_step
    save_top_k=1                                              # Keep only the best checkpoint
    )

    # Callback to save checkpoints every 50 epochs
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath= training_dir + 'periodic/',  # Path to save periodic checkpoints
        filename='epoch-{epoch:02d}',                                        # Save by epoch number
        every_n_epochs=10,                                                   # Save every 50 epochs
        save_top_k=-1                                                        # Save all periodic checkpoints
    )

    # Trainer with multi-GPU setup
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,              # Number of GPUs to use
        strategy='ddp',         # Use Distributed Data Parallel strategy
        precision=32,
        # callbacks=[logger, best_checkpoint_callback, periodic_checkpoint_callback],
        # callbacks=[loss_logger, best_checkpoint_callback],
        callbacks=[loss_logger, best_checkpoint_callback],
        max_epochs=50
    )


    print('Running batch size:', batch_size, 'Learning rate:', learning_rate, 'Resume from:', resume_path)
    # Train!
    # trainer.fit(model, dataloader)
    trainer.fit(model, train_dataloaders=dataloader)


# chmod o+rwx /scratch/YOURNAME/project/ControlNet


# nohup python /scratch/YOURNAME/project/ControlNet/run_vanillaControlnet_train_dlc.py > /scratch/YOURNAME/project/ControlNet_official_data/trainingLog/IJCAI_ControlNet_vanilla/logs/train_tempe.log 2>&1 &

# [1] 3188577: sg236