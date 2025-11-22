from torch import nn

from models.HSLFSR.LFSR.DET.swin_spatial import SpatialTrans
from models.HSLFSR.LFSR.DET.swin_angular_cascaded import CascadedAngTrans
from models.LFSR.det import DET

class DET_AngularSpatialSwin(DET):
    def __init__(self, dim=64, num_heads=4, num_encoders=4, scale_factor=4, window_size=8, m=4):
        super().__init__(dim, num_heads, num_encoders, scale_factor)
        
        ### deep feature extraction
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                SpatialTrans(dim, num_heads, shifted=(i+1)%2==0, window_size=window_size),
                CascadedAngTrans(dim, num_heads, m=m)
            ]) for i in range(num_encoders)
        ])