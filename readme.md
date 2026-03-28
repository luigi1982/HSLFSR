# Hyperspectral Lighttfield Super Resolution

Hyperspectral imaging (HSI) enables remote and non-invasive analysis of material composition
and is therefore used across a wide range of domains. Hyperspectral light-field imaging extends
this capability to snapshot acquisition by capturing spatial and spectral information simultaneously,
making it particularly valuable for time-critical scenarios or continuous monitoring. However,
light-field cameras suffer from an inherent spatial-angular trade-off, finer angular sampling
reduces spatial resolution.
To address this limitation, research in light-field super-resolution (LFSR) leverages complementary
multi-view information, with deep learning methods achieving state-of-the-art performance. 
We investigated whether approaches from LFSR, as well as hyperspectral image superresolution (HSISR) can be effectively adapted to the task of hyperspectral light-field superresolution (HSLFSR).

# Models

This Repository contains multiple Models and Training procedures for Single Image Super Resolution (SISR),
Hyper Spectral Image Super Resolution (HSISR), Light Field Super Resolution (LFSR) and Spectral SuperResolution (SSR)
applying them to Hyper Spectral Light Field Super Resolution (HSLFSR)

### SISR
1. Deep Residual Channel Attention Network (DRCAN)
2. Shifted Window Image Restoration (SWinIR)
3. Hybrid Attention Transformer (HAT)

### HSISR
1. Full 3D U-Net (F3DUN)
2. Spatial–Spectral Aggregation Transformer (SSAformer)

### LFSR
1. Angular Deformable Alignment Module (ADAM)
2. Disentangling (Distg)
3. Epipolar Transformer (EPIT)
4. Disentangling U-Net (Distg UNet)

### SSR
1. Multi-stage Spectral-wise Transformer (MSTpp)

# Training

For training run the **train.py** script in the tasks folder.
Two command line argumnets need to be provided

1. --task, specify the super resolution method, possibler values are 'diff', 'lfsr', 'sisr', to train HSISR or SSR model choose 'lfsr'
2. --config, provide a config file specifying archiotecture and training parameters, examples can be found in the configs directory

# Evaluation

For evaluation run the **evaluate.py** script in the tasks folder.
The same command line arguments need to provided as for running the train.py script.
The experiment under the name specified in the config file is evaluated.

In order to visualize the evaluation run **create_doc.sh** in the result_vis folder.
Two command line arguments need to be provided

1. specify the architecture
2. specify the experiment

# Results