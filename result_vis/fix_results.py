import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from get_results import parse_all

filtered=['distg_unet', 'bicubic']
metrics = ['SSIM', 'SRE', 'SAM', 'PSNR']

parse_all('runs/adam/training/run1-1013-1824')

for model in os.listdir('runs'):
    if model not in filtered:

        for exp in os.listdir('runs/'+model+'/training'):

            path = os.path.join('runs', model, 'training', exp)


            try:
                dfs = parse_all(path)
            except:
                
                dfs = None

            if dfs is not None:
                if 'PSNR Fraunhofer' in list(dfs.keys()):

                    epochs = dfs['PSNR Fraunhofer']['step'].max()
                    writer = SummaryWriter(f'runs/{model}/training/{exp}')

                    for epoch in range(1, epochs+1):

                        avg = dict([(metric, np.zeros(5)) for metric in metrics])
                        for metric in metrics:
                            for i, test_set in enumerate(['EPFL', 'HCI old', 'HCI new', 'Stanford Gantry', 'INRIA Lytro']):
                                avg[metric][i] = dfs[metric+' '+test_set]['value'][epoch-1]

                        writer.add_scalars(
                            'Avg', {'SSIM': avg['SSIM'].mean(), 'PSNR': avg['PSNR'].mean(), 'SAM': avg['SAM'].mean(), 'SRE': avg['SRE'].mean()}, global_step=epoch
                        )