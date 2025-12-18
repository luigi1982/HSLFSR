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

def parse_images(path, prefix):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.IMAGES: 0},
    )
    ea.Reload()

    # list of all image tags
    image_tags = ea.Tags()["images"]

    # select tags like "SSIM_pV/SceneX"
    matched_tags = [t for t in image_tags if t.startswith(prefix)]

    # Load all images per tag
    images = {tag.split('/')[1]: ea.Images(tag) for tag in matched_tags}

    return images

def parse_all(root):
    dfs = {}
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if os.path.isdir(path):
            try:
                scalar = file.split('_')[0]
                df=parse_tensorboard(path, scalar)
                dfs[' '.join(file.split('_'))] = df
            except:
                pass
        else:
            print(path)
            try:
                df=parse_tensorboard(path, 'Train Loss')
                dfs['Train Loss'] = df
            except:
                pass

    return dfs

def parse_all_images(root):

    ssim_pv = parse_images(root, "SSIM_pV")
    psnr_pv = parse_images(root, "PSNR_pV")

    return ssim_pv, psnr_pv

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

