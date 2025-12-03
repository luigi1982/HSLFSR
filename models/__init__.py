### Bicubic
from .utils.commons import BicubicSR

### LFSR models
from .LFSR.epit import EPIT
from .LFSR.lft import LFT
from .LFSR.distg import DISTG
from .LFSR.adam import NET
from .LFSR.distg_unet import DISTG_UNET
from .LFSR.det import DET

### SSR Models
from .SSR.mstpp import MST

### HSISR Models
from .HSISR.f3dun import F3DUN
from .HSISR.ssaformer import SSAFormer

### SISR Models
from .SISR.drcan import DRCAN
from .SISR.hat import HAT
from .SISR.swinir import SwinIR

### Modicfications

#EPIT
from .HSLFSR.LFSR.EPIT.extra_token import EPIT as EPIT_extra_token
from .HSLFSR.LFSR.EPIT.spectral_branch import EPIT as EPIT_spectral_branch
from .HSLFSR.LFSR.EPIT.swin import EPIT_Swin_v1, EPIT_Swin_v2
from .HSLFSR.LFSR.EPIT.spectral_attention import EPIT as EPIT_spectral_attention
from .HSLFSR.LFSR.EPIT.hfa import EPIT as EPIT_hfa

#DET
from .HSLFSR.LFSR.DET.ablation_angular import DET as DET_ablation_angular
from .HSLFSR.LFSR.DET.swin_angular import DET as DET_swin_angular
from .HSLFSR.LFSR.DET.swin_angular_cascaded import DET as DET_swin_angular_cascaded
from .HSLFSR.LFSR.DET.swin_spatial import DET as DET_swin_spatial
from .HSLFSR.LFSR.DET.swin_angular_spatial import DET as DET_swin_angular_spatial

#DISTG UNET
from .HSLFSR.LFSR.UNET.f3dun import F3DUN_UNET

#DRCAN
from .HSLFSR.SISR.DRCAN.drcan_concat import DRCAN_concat
from .HSLFSR.SISR.DRCAN.drcan_film import DRCAN_FiLM

#F3DUN
from .HSLFSR.HSISR.F3DUN.f4dun import F4DUN

MODEL_REGISTRY = {
    
    "bicubic": BicubicSR,
    "epit": EPIT,
    "lft": LFT,
    "det": DET,
    "distg": DISTG,
    "distg_unet": DISTG_UNET,
    "adam": NET,
    "mst": MST,
    "f3dun": F3DUN,
    "ssaformer": SSAFormer,
    "drcan": DRCAN,
    "hat": HAT,
    "swinir": SwinIR,

    ### modifications

    #EPIT
    "epit_extra_token": EPIT_extra_token,
    "epit_swin_v1": EPIT_Swin_v1,
    "epit_swin_v2": EPIT_Swin_v2,
    "epit_spectral_branch": EPIT_spectral_branch,
    "epit_spectral_attention": EPIT_spectral_attention,
    "epit_hfa": EPIT_hfa,

    #DET
    "det_ablation_angular": DET_ablation_angular,
    "det_swin_angular": DET_swin_angular,
    "det_swin_angular_cascaded": DET_swin_angular_cascaded,
    "det_swin_spatial": DET_swin_spatial,
    "det_swin_angular_spatial": DET_swin_angular_spatial,

    #DISTG UNET
    "unet_f3dun": F3DUN_UNET,

    #DRCAN
    "drcan_concat": DRCAN_concat,
    "drcan_film": DRCAN_FiLM,

    #F3DUN
    "f3dun_f4dun": F4DUN,
}