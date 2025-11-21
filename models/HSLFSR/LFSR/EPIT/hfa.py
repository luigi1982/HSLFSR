from torch import nn
from einops import rearrange

from models.LFSR.epit import Residual, AltFilter, SpatialConv
from models.LFSR.det import FeatureAggregation

class EPIT(nn.Module):
    def __init__(self, channels, ang_res=5, use_as_encoder=False, number_alt_filters=4):
        super().__init__()

        self.ang_res = ang_res
        self.use_as_encoder = use_as_encoder

        self.in_conv = nn.Sequential(
            nn.Conv3d(1, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            Residual(SpatialConv(channels))
        )

        self.EPI_features = nn.ModuleList(
            [AltFilter(channels) for _ in range(number_alt_filters)]
        )

        self.hfa = FeatureAggregation(channels, num_encoders=number_alt_filters)

        channels = int(channels*number_alt_filters/2)
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*16, 1, bias=False),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):
        [_, _, h, w] = x.size()
        x = self.in_conv(x.unsqueeze(1))
        fs = []
        for filter in self.EPI_features:
            x = filter(x)
            fs.append(x)

        # feature aggregation
        x = self.hfa(fs)

        # x is shape B x C x UV x H x W
        # reshape to B x C x UV x HW for upsampling
        x_up = rearrange(x, 'b c (u v) h w -> b c (u h) (v w)', u=self.ang_res, v=self.ang_res)
        x_up = self.upsampling(x_up)
        x_up = rearrange(x_up, 'b c (u h) (v w) -> b c (u v) h w', h=4*h, w=4*w, u=self.ang_res, v=self.ang_res)

        if self.use_as_encoder:
            return x_up, rearrange(x, 'b c (u v) h w -> b c (h u) (w v)', u=self.ang_res, v=self.ang_res)
        else:
            x_up = x_up.view((-1, 25, 128, 128))
            return x_up