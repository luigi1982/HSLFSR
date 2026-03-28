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

![alt text](https://github.com/luigi1982/HSLFSR/blob/showcase/images/example.png)

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

1. --task, specify the super resolution method, possibler values are 'diff', 'lfsr', 'sisr', to train HSISR or SSR models choose 'lfsr'
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

In the following table we display the results for Bicubic interpolation, EPIT, F3DUN and Distg UNet.
We use the four metrics Peak Signal to Noise Ratio (PSNR), Structural Similarity Index (SSIM), Spectral Angular Map (SAM) and Spectral Relatic Error (SRE).
In the right part of the table we also list the number of parameters, number of floating point operations and time required for the forward pass in ms.

![alt text](https://github.com/luigi1982/HSLFSR/blob/showcase/images/results_vanilla.png)

A problem we encountered is that in scenes lit by artificial lighting,
the channels capturing the wavelengths towards the extremes and outside of the visable spectrum have low signal and turn out dark.

These inflate results.
We capture a set of images with multiple exposure times,
to obtain more uniform signal strengths across all channels.

The top plot showcases PSNR results pre channel for an images recorded with a single exposure time,
bottom plot for multiple different exposures. 


![alt text](https://github.com/luigi1982/HSLFSR/blob/showcase/images/results_vanilla_disected.png)

Qualitative results.

![alt text](https://github.com/luigi1982/HSLFSR/blob/showcase/images/qual_result.png)

We repeat the first experiment for multiple modifications of the EPIT architecture. We assess
four options to decrease computational load. First, we simply half the channel dimension from
64 to 32, we denote the model by EPIT rcd, where rcd is short for reduced channel dimension.
Secondly, we reduce the number of Blocks from 5 to 3, the model is denoted by EPIT rbn,
short for reduced block number. Lastly, we compare to that the performance of our proposed
architecture, integrating the shifted window mechanism into the EPIT model. It is denoted by
EPIT SWin. In order to ablate, wether the shift leads to improvement, we also train a model
for which only windowed attention is employed, but the cyclic shift is omitted. The model is
denoted by EPIT Win.

![alt text](https://github.com/luigi1982/HSLFSR/blob/showcase/images/changes.png)