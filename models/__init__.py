### LFSR models
from .LFSR.epit import EPIT
from .LFSR.lft import LFT
from .LFSR.distg import DISTG
from .LFSR.adam import NET
from .LFSR.distg_unet import DISTG_UNET

### SSR Models
from .SSR.mstpp import MST

### HSISR Models
from .HSISR.f3dun import F3DUN
from .HSISR.ssaformer import SSAFormer

### SISR Models
from .SISR.drcan import DRCAN
from .SISR.hat import HAT
from .SISR.swinir import SwinIR

MODEL_REGISTRY = {
    "epit": EPIT,
    "lft": LFT,
    "distg": DISTG,
    "distg_unet": DISTG_UNET,
    "adam": NET,
    "mst": MST,
    "f3dun": F3DUN,
    "ssaformer": SSAFormer,
    "drcan": DRCAN,
    "hat": HAT,
    "swinir": SwinIR
}