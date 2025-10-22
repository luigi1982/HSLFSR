import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os
import h5py
import numpy as np
import yaml
from jsonargparse import namespace_to_dict

from dataset import LightFieldDataset, LightFieldTestDataset
from utils.utils import *
from utils.metrics import *

from models import MODEL_REGISTRY

### parsing of config
from jsonargparse import ArgumentParser, ActionConfigFile
from config import TrainConfig, make_serializable

parser = ArgumentParser(description="Training script with config files")
parser.add_argument("--config", action=ActionConfigFile)
parser.add_class_arguments(TrainConfig, nested_key="train")
cfg = parser.parse_args()

EPOCHS=cfg.train.epochs
BS=cfg.train.batch_size
DEVICE=cfg.train.device
MODEL=cfg.train.model.model
EVAL_BS=cfg.train.evaluation.batch_size

### Get the Data

#transformation

## mean and std of dataset
mean = (0.0403, 0.0527, 0.0594, 0.0673, 0.0781, 0.0874, 0.0926, 0.0936, 0.0957,
        0.0957, 0.0957, 0.0961, 0.1002, 0.1054, 0.1089, 0.1089, 0.1088, 0.1069,
        0.1054, 0.1056, 0.1097, 0.1130, 0.1148, 0.1158, 0.1180)
std = (0.0523, 0.0696, 0.0780, 0.0878, 0.1010, 0.1117, 0.1169, 0.1175, 0.1177,
        0.1156, 0.1132, 0.1125, 0.1165, 0.1242, 0.1319, 0.1352, 0.1343, 0.1317,
        0.1293, 0.1279, 0.1298, 0.1315, 0.1315, 0.1316, 0.1333)

## mean and std of dataset after normlaizing data with the above mean and std
## and then clamping the values to [-1, +1]
mean2 = (-0.1506, -0.1519, -0.1525, -0.1528, -0.1532, -0.1527, -0.1520, -0.1518,
        -0.1520, -0.1503, -0.1483, -0.1491, -0.1503, -0.1501, -0.1490, -0.1475,
        -0.1460, -0.1457, -0.1448, -0.1446, -0.1449, -0.1449, -0.1448, -0.1456,
        -0.1466)
std2 = (0.6029, 0.6081, 0.6144, 0.6195, 0.6274, 0.6305, 0.6338, 0.6341, 0.6425,
        0.6454, 0.6520, 0.6535, 0.6525, 0.6396, 0.6233, 0.6111, 0.6134, 0.6140,
        0.6173, 0.6241, 0.6358, 0.6480, 0.6595, 0.6665, 0.6722)

transform = transforms.Compose([
    transforms.Normalize(mean, std),
    transforms.Lambda(lambda x: torch.clamp(x, -1, 1)),
    transforms.Normalize(mean2, std2)
])

data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']

for data_set in data_list:

    data_sets = [data for data in data_list if data != data_set]
    test_set = data_set

    #load the datasets in data_sets as training data
    #load the data in test_set as test data

    