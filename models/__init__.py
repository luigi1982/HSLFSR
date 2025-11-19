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
from .HSLFSR.LFSR.EPIT.swin import EPIT as EPIT_swin
from .HSLFSR.LFSR.EPIT.spectral_attention import EPIT as EPIT_spectral_attention

#DRCAN
from .HSLFSR.SISR.DRCAN.drcan_concat import DRCAN_concat
from .HSLFSR.SISR.DRCAN.drcan_film import DRCAN_FiLM

MODEL_REGISTRY = {
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
    "epit_swin": EPIT_swin,
    "epit_spectral_branch": EPIT_spectral_branch,
    "epit_spectral_attention": EPIT_spectral_attention,

    #DRCAN
    "drcan_concat": DRCAN_concat,
    "drcan_film": DRCAN_FiLM
}