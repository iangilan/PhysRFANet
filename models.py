import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_blocks import unet_down, unet_mid, unet_up, self_attention

class RFACNN(nn.Module):
    def __init__(self):
        super(RFACNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=(1, 1, 1))
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, Ninput, MR):
        x = torch.cat([Ninput, MR], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class RFAUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, in_inter=True, out_inter=True):
        super(RFAUNet, self).__init__()
        
        self.in_inter = in_inter
        self.out_inter = out_inter

        # Initialize layers using functions
        self.down1 = unet_down(in_channels, 64, normalize=False)
        self.down2 = unet_down(64, 128)
        self.down3 = unet_down(128, 256)
        
        self.mid1 = unet_mid(256, 256)
        self.mid2 = unet_mid(256, 256)
        self.mid3 = unet_mid(256, 256)
        self.mid4 = unet_mid(256, 256)
        
        self.up1 = unet_up(256, 128)
        self.up2 = unet_up(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 2, 2)
        )

    def forward(self, Ninput, MR):
        x = torch.cat([Ninput, MR], dim=1)
        
        if self.in_inter:
            x = F.interpolate(x, [40, 40, 40])
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        m1 = self.mid1(d3, d3)
        m2 = self.mid2(m1, m1)
        m3 = self.mid3(m2, m2)
        m4 = self.mid4(m3, m3)
        
        u1 = self.up1(m4, d2)
        u2 = self.up2(u1, d1)
        u3 = self.final(u2)
        
        if self.out_inter:
            u3 = F.interpolate(u3, [41, 41, 41])
            
        return u3

class RFAAttUNet(RFAUNet):
    def __init__(self, in_channels=2, out_channels=1, in_inter=True, out_inter=True):
        super().__init__(in_channels, out_channels, in_inter, out_inter)

        self.in_inter = in_inter
        self.out_inter = out_inter

        self.sa1 = self_attention(256)
        self.sa2 = self_attention(256)
        self.sa3 = self_attention(128)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 2, 2)
        )

    def forward(self, Ninput, MR):
        x = torch.cat([Ninput, MR], dim=1)

        if self.in_inter:
            x = F.interpolate(x, [40, 40, 40])

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        m1 = self.mid1(d3)
        m2 = self.mid2(m1)
        m3 = self.mid3(m2)
        m4 = self.mid4(m3)
        m4 = self.sa1(m4)
        
        u1 = self.up1(m4)
        u1 = self.sa2(u1)
        u2 = self.up2(u1)
        u2 = self.sa3(u2)
        u3 = self.final(u2)

        if self.out_inter:
            u3 = F.interpolate(u3, [41, 41, 41])

        return u3