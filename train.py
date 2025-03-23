import yaml
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import model3
from dataloader import DataLoader, AudioTextDataset
from pytorch_lightning.callbacks import LearningRateMonitor
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda")




# Load configuration
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize Weights & Biases
wandb_logger = loggers.WandbLogger(
    name="LAS_Model_scalerGan_small_range",
    project=config['wandb']['project_name'],
    entity=config['wandb']['entity']
)

# Define the PyTorch Lightning model
# las_model = model.LASModel(input_dim=80, hidden_dim=50, vocab_size=30522)

# Model instantiation
las_model = model3.LASModel(
    input_dim=config['training']["input_dim"],
    hidden_dim=config['training']["hidden_dim"],
    vocab_size=config['training']["vocab_size"],
    num_lstm_layers=config['training']["num_lstm_layers"],
    projection_dim=config['training']["projection_dim"],
    channels=config['training']["channels"],
    learning_rate=config['training']["learning_rate"],
    teacher_forcing=config['training']["teacher_forcing"]
).to(device)

# DataLoaders with drop_last=True to maintain consistency in batch sizes
train_loader = DataLoader(
    AudioTextDataset(audio_paths_file=config['training']['data_path'], config=config['audio_processing'], train = True),
    batch_size=config['training']['batch_size'],
    shuffle=True,
    drop_last=True,
    pin_memory=True,  # Helps when using GPU
    num_workers=config['training']['num_workers'],
)
val_loader = DataLoader(
    AudioTextDataset(audio_paths_file=config['validation']['data_path'], config=config['audio_processing'],train = False),
    batch_size=config['training']['batch_size'],
    drop_last=True,
    pin_memory=True,  # Helps when using GPU
    num_workers=config['training']['num_workers'],
)
test_loader = DataLoader(
    AudioTextDataset(audio_paths_file=config['test']['data_path'], config=config['audio_processing'],train = False),
    batch_size=config['test']['batch_size'],
    drop_last=True,
    pin_memory=True,  # Helps when using GPU
    num_workers=config['training']['num_workers'],
)

# Callbacks for monitoring training
checkpoint_callback = ModelCheckpoint(
    dirpath= config['training']['checkpoint'],
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

# Callback to monitor the learning rate
lr_monitor = LearningRateMonitor(logging_interval=None)  # or 'epoch' if you prefer less frequent updates
# Trainer setup with Distributed Data Parallel strategy
trainer = Trainer(
    log_every_n_steps = 1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback,lr_monitor],
    max_epochs=config['training']['epochs'],
    devices=config['training']['num_gpus'],
    accelerator='gpu',
    strategy=DDPStrategy(find_unused_parameters=True),
    check_val_every_n_epoch=10
)

#Start training and testing
checkpoint_path = '/home/galraviv/scaler_gan/scaler_gan/LAS_model/specAug_n/epoch=69-step=12460.ckpt'
#'/home/galraviv/scaler_gan/scaler_gan/LAS_model/checkpoints_scalerGan/epoch=59-step=10680.ckpt'
#'/home/galraviv/scaler_gan/scaler_gan/LAS_model/checkpoints_best_run/epoch=64-step=5785.ckpt'
#'/home/galraviv/scaler_gan/scaler_gan/LAS_model/checkpoints_specAug/epoch=79-step=8080.ckpt'
#'/home/galraviv/scaler_gan/scaler_gan/LAS_model/checkpoints_batch_128_no_teacher_forcing/epoch=14-step=330.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)  # Ensure the checkpoint is loaded to the correct device
las_model.load_state_dict(checkpoint['state_dict'],strict=True)
#trainer.fit(las_model, train_loader, val_loader)
trainer.test(las_model,dataloaders=test_loader)

