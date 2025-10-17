from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

def parse_tensorboard(path, scalar):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    return pd.DataFrame(ea.Scalars(scalar))

def parse_all(root):
    dfs = {}
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if os.path.isdir(path):
            scalar = file.split('_')[0]
            df=parse_tensorboard(path, scalar)
            dfs[' '.join(file.split('_'))] = df
        else:
            try:
                df=parse_tensorboard(path, 'Train Loss')
                dfs['Train Loss'] = df
            except:
                pass

    return dfs

def parse_avgs(root):
    dfs = {}
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if os.path.isdir(path):
            if 'Avg' in file:
                scalar = file.split('_')[0]
                df=parse_tensorboard(path, scalar)
                dfs[' '.join(file.split('_'))] = df
        else:
            try:
                df=parse_tensorboard(path, 'Train Loss')
                dfs['Train Loss'] = df
            except:
                pass

    return dfs

