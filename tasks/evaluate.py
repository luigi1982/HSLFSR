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
parser.add_argument("--model", default=None)
parser.add_argument("--config", action=ActionConfigFile)
parser.add_argument("--experiment", default=False)
parser.add_class_arguments(TrainConfig, nested_key="train")
cfg = parser.parse_args()

EPOCHS=cfg.train.epochs
BS=cfg.train.batch_size
DEVICE=cfg.train.device
MODEL=cfg.model
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
train_data_list = ['Lab_day']
test_data_list = ['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec', 'Texts']


if cfg.task == 'diff':

    #load the model
    denoise_fn = MODEL_REGISTRY['distg_unet'](32, 32, 32)
    encoder_fn = MODEL_REGISTRY['epit'](32, use_as_encoder=True)

    epoch = max([int(net.split('_')[-1].split('.')[0]) for net in os.listdir(f'models_/{MODEL}/training/{cfg.experiment}')])
    checkpoint = f'models_/{MODEL}/training/{cfg.experiment}/net_epoch_{epoch}.pth'
    EXP_NAME = cfg.experiment

    #instatiate the trainer class
    trainer = trainer(
        EXP_NAME, MODEL,
        denoise_fn, encoder_fn,
        train_data_list, test_data_list,
        EPOCHS, DEVICE, BS,
        EVAL_BS, EVAL_STEP, SAVE_LF_STEP,
        lr=LR, lr_decay_steps=LR_DECAY_STEP, gamma=GAMMA,
        start_epoch=epoch,
        mode='evaluate',
        checkpoint=checkpoint
    )
    
else:

    #load the model
    print(cfg.train.model.dim)
    model = MODEL_REGISTRY[MODEL](cfg.train.model.dim)

    #load the specified checkpoint
    if MODEL != 'bicubic':
        epoch = max([int(net.split('_')[-1].split('.')[0]) for net in os.listdir(f'models_/{MODEL}/training/{cfg.experiment}')])
        checkpoint = f'models_/{MODEL}/training/{cfg.experiment}/net_epoch_{epoch}.pth'
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        epoch = 0
    EXP_NAME = cfg.experiment


    #instatiate the trainer class
    trainer = trainer(
        EXP_NAME, MODEL, model, 
        train_data_list, test_data_list,
        EPOCHS, DEVICE, BS,
        EVAL_BS, EVAL_STEP, SAVE_LF_STEP,
        lr=LR, lr_decay_steps=LR_DECAY_STEP, gamma=GAMMA,
        start_epoch=epoch,
        mode='evaluate'
    )

#evaluate on seven test sets with bicubuc downsampling
print('Evaluating on bicubically downsampled data')
trainer.evaluate(0, True, save_model=False, save_pV=True, 
                 test_sets=['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec']
                 )

#evaluate on seven test sets with classical degradation scheme
print('Evaluating on classical degradation scheme')
trainer.evaluate(0, True, save_model=False, save_pV=True, degradation_process='classical',
                 test_sets=['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec']
    )

#evaluate on the Text dataset
print('Evaluating no degradation')
trainer.evaluate(0, True, save_model=False, save_pV=True, degradation_process='id', track_metrics=False, test_sets=['Texts'])