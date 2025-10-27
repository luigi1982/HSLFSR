import importlib
import os
import sys
from datetime import datetime

sys.path.insert(1, os.path.join(sys.path[0], '..'))

### parsing of config
from jsonargparse import ArgumentParser, ActionConfigFile
from config import TrainConfig, make_serializable

parser = ArgumentParser(description="Training script with config files")
parser.add_argument("--task", default='lfsr')
parser.add_argument("--config", action=ActionConfigFile)
parser.add_class_arguments(TrainConfig, nested_key="train")
cfg = parser.parse_args()

EPOCHS=cfg.train.epochs
BS=cfg.train.batch_size
DEVICE=cfg.train.device
MODEL=cfg.train.model.model
EXP_NAME = cfg.train.name
EVAL_BS=cfg.train.evaluation.batch_size
EVAL_STEP=cfg.train.evaluation.eval_step
SAVE_LF_STEP=cfg.train.evaluation.save_lf_step
LR = cfg.train.optim.lr
LR_DECAY_STEP = cfg.train.optim.lr_decay_steps
GAMMA = cfg.train.optim.gamma

from models import MODEL_REGISTRY

#load the model
model = MODEL_REGISTRY[MODEL](cfg.train.model.dim)

#load the trainer
pkg = importlib.import_module(cfg.task)
cls_name = cfg.task.upper() + 'Trainer'
trainer = getattr(pkg, cls_name)

#load the train and test data
data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']

#get the date
now = datetime.now()
now = now.strftime('%m%d-%H%M')

for i, data_set in enumerate(data_list):

        print(f'[{i+1}/{len(data_list)}] Hold Out: {data_set}')

        train_data_list = [data for data in data_list if data != data_set]
        test_data_list = [data_set+'-hold_out', 'EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']

        #load the datasets in data_sets as training data
        #load the data in test_set as test data

        exp_name = 'hold_out='+data_set

        trainer_instance = trainer(
                exp_name, MODEL, model, 
                train_data_list, test_data_list,
                EPOCHS, DEVICE, BS,
                EVAL_BS, EVAL_STEP, SAVE_LF_STEP,
                lr=LR, lr_decay_steps=LR_DECAY_STEP, gamma=GAMMA,
                cross_val_run=EXP_NAME+'-'+now
        )

        #start training
        trainer_instance.training()

    