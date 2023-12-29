import torch
import torch.nn as nn

'''
These are the building blocks of the UNet and the AttentionUNet.
'''

class unet_down(nn.Module):
    
    def __init__(self, in_channels, out_channels, normalize=True):
        super(unet_down, self).__init__()
        layers = [nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
class unet_mid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unet_mid, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, skip_x):
        x = self.model(x)
        x = torch.add(x, skip_x)
        return x
    
class unet_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unet_up, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_x):
        x = self.model(x)
        x = torch.cat((x, skip_x), 1)
        return x
    
class self_attention(nn.Module):
    def __init__(self, channels):
        super(self_attention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).contiguous().view(-1, self.channels, size, size, size)