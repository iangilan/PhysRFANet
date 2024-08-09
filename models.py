import torch
import torch.nn as nn
import torch.nn.functional as F

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
       
        self.skipMid = self.skip_mid
        self.skipUp = self.skip_up
       
        # Initialize layers using functions
        self.down1 = self.unet_down(in_channels, 64, normalize=False)
        self.down2 = self.unet_down(64, 128)
        self.down3 = self.unet_down(128, 256)
       
        self.mid1 = self.unet_mid(256, 256, dropout=0)
        self.mid2 = self.unet_mid(256, 256, dropout=0)
        self.mid3 = self.unet_mid(256, 256, dropout=0)
        self.mid4 = self.unet_mid(256, 256, dropout=0)
       
        self.up1 = self.unet_up(256, 128)
        self.up2 = self.unet_up(256, 64)
       
        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 2, 2)
        )

    @staticmethod
    def unet_down(in_size, out_size, normalize=True, dropout=0.0):
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def unet_mid(in_size, out_size, dropout=0.0):
        layers = [
            nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
           
        return nn.Sequential(*layers)

    @staticmethod
    def unet_up(in_size, out_size, dropout=0.0):
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 2, 2, bias=False),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)
   
    def skip_mid(self,x,skip_input):
        x = torch.add(x, skip_input)
        return x
       
    def skip_up(self,x,skip_input):
        x = torch.cat((x, skip_input), 1)
        return x
   
   
    def forward(self, Ninput, MR):
        x = torch.cat([Ninput, MR], dim=1)
       
        if self.in_inter:
            x = F.interpolate(x, [40, 40, 40])
       
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
       
        m1 = self.mid1(d3)
        m1 = self.skipMid(m1,d3)
        m2 = self.mid2(m1)
        m2 = self.skipMid(m2,m1)
        m3 = self.mid3(m2)
        m3 = self.skipMid(m3,m2)
        m4 = self.mid4(m3)
        m4 = self.skipMid(m4,m3)
       
        u1 = self.up1(m4)
        u1 = self.skipUp(u1,d2)
        u2 = self.up2(u1)
        u2 = self.skipUp(u2,d1)
        u3 = self.final(u2)
       
        if self.out_inter:
            u3 = F.interpolate(u3, [41, 41, 41])

        return u3 

class RFAAttUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, in_inter=True, out_inter=True):
        super(RFAAttUNet, self).__init__()

        self.in_inter = in_inter
        self.out_inter = out_inter
       
        self.skipMid = self.skip_mid
        self.skipUp = self.skip_up
       
        self.down1 = self.unet_down(in_channels, 64, normalize=False)
        self.down2 = self.unet_down(64, 128)
        self.down3 = self.unet_down(128, 256)

        self.mid1 = self.unet_mid(256, 256, dropout=0)
        self.mid2 = self.unet_mid(256, 256, dropout=0)
        self.mid3 = self.unet_mid(256, 256, dropout=0)
        self.mid4 = self.unet_mid(256, 256, dropout=0)

        self.sa1 = self.self_attention(256, num_heads=8)
        self.sa2 = self.self_attention(256, num_heads=8)
        #self.sa3 = self.self_attention(128, num_heads=8)

        self.up1 = self.unet_up(256, 128)
        self.up2 = self.unet_up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 2, 2)
        )

    @staticmethod
    def unet_down(in_size, out_size, normalize=True, dropout=0.0):
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def unet_mid(in_size, out_size, dropout=0.0):
        layers = [
            nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def unet_up(in_size, out_size, dropout=0.1):
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 2, 2, bias=False),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    @staticmethod
    def self_attention(channels, num_heads):
        mha = nn.MultiheadAttention(channels, num_heads, batch_first=True) #num_heads=8
        ln = nn.LayerNorm(channels)
        ff_self = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        return nn.ModuleList([mha, ln, ff_self])

    def apply_attention(self, attention_layers, x):
        mha, ln, ff_self = attention_layers
        B, C, D, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        x_ln = ln(x)
        attn_output, _ = mha(x_ln, x_ln, x_ln)
        x = attn_output + x
        x = ff_self(x) + x
        x = x.permute(0, 2, 1).view(B, C, D, H, W)
        return x
   
    def skip_mid(self,x,skip_input):
        x = torch.add(x, skip_input)
        return x
   
    def skip_up(self, x, skip_input):
        return torch.cat((x, skip_input), dim=1)

    def forward(self, Ninput, MR):
        x = torch.cat([Ninput, MR], dim=1)

        if self.in_inter:
            x = F.interpolate(x, [40, 40, 40])

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        m1 = self.mid1(d3)
        m1 = self.skipMid(m1,d3)
        m2 = self.mid2(m1)
        m2 = self.skipMid(m2,m1)
        m3 = self.mid3(m2)
        m3 = self.skipMid(m3,m2)
        m4 = self.mid4(m3)
        m4 = self.skipMid(m4,m3)

        m4 = self.apply_attention(self.sa1, m4)
        u1 = self.up1(m4)
        u1 = self.skip_up(u1, d2)
        u1 = self.apply_attention(self.sa2, u1)
        u2 = self.up2(u1)
        u2 = self.skip_up(u2, d1)
        #u2 = self.apply_attention(self.sa3, u2)
        u3 = self.final(u2)

        if self.out_inter:
            u3 = F.interpolate(u3, [41, 41, 41])
        return u3

'''
class RFAAttUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, in_inter=True, out_inter=True):
        super(RFAAttUNet, self).__init__()

        self.in_inter = in_inter
        self.out_inter = out_inter

        self.down1 = self.unet_down(in_channels, 64, normalize=False)
        self.down2 = self.unet_down(64, 128)
        self.down3 = self.unet_down(128, 256)

        self.mid1 = self.unet_mid(256, 256, dropout=0)
        self.mid2 = self.unet_mid(256, 256, dropout=0)
        self.mid3 = self.unet_mid(256, 256, dropout=0)
        self.mid4 = self.unet_mid(256, 256, dropout=0)

        self.sa1 = self.self_attention(256, num_heads=8)
        self.sa2 = self.self_attention(256, num_heads=8)
        #self.sa3 = self.self_attention(128, num_heads=8)

        self.up1 = self.unet_up(256, 128)
        self.up2 = self.unet_up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 2, 2)
        )

    @staticmethod
    def unet_down(in_size, out_size, normalize=True, dropout=0.0):
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def unet_mid(in_size, out_size, dropout=0.0):
        layers = [
            nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def unet_up(in_size, out_size, dropout=0.0):
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 2, 2, bias=False),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    @staticmethod
    def self_attention(channels, num_heads):
        mha = nn.MultiheadAttention(channels, num_heads, batch_first=True) #num_heads=8
        ln = nn.LayerNorm(channels)
        ff_self = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        return nn.ModuleList([mha, ln, ff_self])

    def apply_attention(self, attention_layers, x):
        mha, ln, ff_self = attention_layers
        B, C, D, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        x_ln = ln(x)
        attn_output, _ = mha(x_ln, x_ln, x_ln)
        x = attn_output + x
        x = ff_self(x) + x
        x = x.permute(0, 2, 1).view(B, C, D, H, W)
        return x

    def skip_up(self, x, skip_input):
        return torch.cat((x, skip_input), dim=1)

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

        m4 = self.apply_attention(self.sa1, m4)
        u1 = self.up1(m4)
        u1 = self.skip_up(u1, d2)
        u1 = self.apply_attention(self.sa2, u1)
        u2 = self.up2(u1)
        u2 = self.skip_up(u2, d1)
        #u2 = self.apply_attention(self.sa3, u2)
        u3 = self.final(u2)

        if self.out_inter:
            u3 = F.interpolate(u3, [41, 41, 41])
        return u3
'''
