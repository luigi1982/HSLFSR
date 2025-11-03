import argparse
import os
import h5py
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_angRes", type=int, default=5, help="angular resolution")
    parser.add_argument('--data_for', type=str, default='training_hsi', help='')
    parser.add_argument('--src_data_path', type=str, default='../datasets/', help='')
    parser.add_argument('--save_data_path', type=str, default='./', help='')
    parser.add_argument('--data_set', type=str, default='', help='')
    return parser.parse_args()


def main(args):
    
    data_dir = args.src_data_path
    data_for = args.data_for
    datasets = ['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors']

    patch_size_LR = 32
    patch_size_HR = 4 * patch_size_LR
    stride = patch_size_LR * 2

    for data in datasets:
        path = os.path.join(data_dir, data, data_for)

        num_patches = 0

        save_dir = os.path.join('data/training', data)
        os.makedirs(save_dir, exist_ok=True)

        for file in os.listdir(path):

            scene_name = os.path.splitext(file)[0]
            
            #load the lf
            file_path = os.path.join(path, file)
            with h5py.File(file_path, 'r') as hf:
                lf = np.array(hf['HR'])

            #extract pathches from lightfield

            n, h, w = lf.shape
            h -= (h - patch_size_HR)%stride
            w -= (w - patch_size_HR)%stride

            n = (h - patch_size_HR)//stride
            m = (w - patch_size_HR)//stride

            for i in range(n+1):
                for j in range(m+1):

                    x = i*stride
                    y = j*stride

                    patch = lf[:, x: x + patch_size_HR, y: y+patch_size_HR]
                    assert(patch.shape == (25, patch_size_HR, patch_size_HR))
                    file_name = os.path.join(save_dir, f'{scene_name}_{i}-{j}')
                    with h5py.File(file_name, 'w') as hf:
                        hf.create_dataset('LF', data=patch, dtype='single')
                        hf.close()

                    num_patches += 1

        print(f'created {num_patches} from {data}')

        

if __name__ == '__main__':
    args = parse_args()
    main(args)