import torch

import importlib
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

### parsing of config
from jsonargparse import ArgumentParser, ActionConfigFile
from config import TrainConfig, make_serializable

parser = ArgumentParser(description="Training script with config files")
parser.add_argument("--task", default='lfsr')
parser.add_argument("--config", action=ActionConfigFile)
parser.add_argument("--from_checkpoint", default=False)
parser.add_argument("--modification", default=None)
parser.add_class_arguments(TrainConfig, nested_key="train")
cfg = parser.parse_args()

EPOCHS=cfg.train.epochs
BS=cfg.train.batch_size
DEVICE=cfg.train.device
MODEL=cfg.modification if cfg.modification else cfg.train.model.model
EXP_NAME = cfg.train.name
EVAL_BS=cfg.train.evaluation.batch_size
EVAL_STEP=cfg.train.evaluation.eval_step
SAVE_LF_STEP=cfg.train.evaluation.save_lf_step
LR = cfg.train.optim.lr
LR_DECAY_STEP = cfg.train.optim.lr_decay_steps
GAMMA = cfg.train.optim.gamma

from models import MODEL_REGISTRY

#load the trainer
pkg = importlib.import_module(cfg.task)
cls_name = cfg.task.upper() + 'Trainer'
trainer = getattr(pkg, cls_name)

#load the train and test data
train_data_list = ['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors']
test_data_list = ['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors']


if cfg.task == 'diff':

    #load the model
    denoise_fn = MODEL_REGISTRY['distg_unet'](32, 32, 32)
    encoder_fn = MODEL_REGISTRY['epit'](32, use_as_encoder=True)

    #load encoder model
    ENC_PATH='models_/epit/training/dim=32-1114-1821/net_epoch_80.pth'
    encoder_fn.load_state_dict(torch.load(ENC_PATH, weights_only=True))

    #instatiate the trainer class
    trainer = trainer(
        EXP_NAME, MODEL,
        denoise_fn, encoder_fn, 
        train_data_list, test_data_list,
        EPOCHS, DEVICE, BS,
        EVAL_BS, EVAL_STEP, SAVE_LF_STEP,
        lr=LR, lr_decay_steps=LR_DECAY_STEP, gamma=GAMMA
    )
    
else:

    #load the model
    model = MODEL_REGISTRY[MODEL](cfg.train.model.dim)

    #if continue tranining from a checkpoint
    if cfg.from_checkpoint:
        epoch = max([int(net.split('_')[-1].split('.')[0]) for net in os.listdir(f'models_/{MODEL}/training/{cfg.from_checkpoint}')])
        checkpoint = f'models_/{MODEL}/training/{cfg.from_checkpoint}/net_epoch_{epoch}.pth'
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        EXP_NAME = cfg.from_checkpoint
    else:
        epoch=0 

    #instatiate the trainer class
    trainer = trainer(
        EXP_NAME, MODEL, model, 
        train_data_list, test_data_list,
        EPOCHS, DEVICE, BS,
        EVAL_BS, EVAL_STEP, SAVE_LF_STEP,
        lr=LR, lr_decay_steps=LR_DECAY_STEP, gamma=GAMMA,
        start_epoch=epoch
    )

#start training
trainer.training()
