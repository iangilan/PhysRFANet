import torch
import torch.nn as nn
from utils import conv_block, up_conv, Attention_block

class TemperatureCNN(nn.Module):
    def __init__(self):
        super(TemperatureCNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=2, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
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


class TemperatureUNet(nn.Module):
    def __init__(self):
        super(TemperatureUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(2, 32, 3, 1, pool=False)#True)
        self.enc2 = self.conv_block(32, 64, 3, 2, pool=False)#True)
        self.enc3 = self.conv_block(64, 128, 3, 2, pool=False)#True)
        self.enc4 = self.conv_block(128, 256, 3, 2, pool=False)#True)
        self.enc5 = self.conv_block(256, 512, 3, 2, pool=False)#True) 

        # Decoder with skip connections       
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.br1 = self.BR_block(256)  
        
        self.up2 = nn.ConvTranspose3d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.br2 = self.BR_block(128)     

        self.up3 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.br3 = self.BR_block(64)          

        self.up4 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.br4 = self.BR_block(32)         

        self.up5 = nn.ConvTranspose3d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.br5 = self.BR_block(16)           

        self.final = nn.Conv3d(16, 1, kernel_size=3, stride=2, padding=1)

    def BR_block(self, out_channels):
        layers = [
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def conv_block(self, in_channels, out_channels, kernel, stride, pool=True):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool3d((2, 2, 2), stride=2))
        #layers.append(nn.Dropout(0.0))  # Dropout layer for regularization
        return nn.Sequential(*layers)

        # Weight initialization as before
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)    
    
    def forward(self, Ninput, MR):
        # Concatenate the inputs along the channel dimension
        x = torch.cat([Ninput, MR], dim=1)        

        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        up1_out = self.up1(enc5_out)
        up1_out = self.br1(up1_out)
        dec1_out = up1_out
        
        up2_out = self.up2(torch.cat([dec1_out, enc4_out], 1))
        up2_out = self.br2(up2_out)
        dec2_out = up2_out

        up3_out = self.up3(torch.cat([dec2_out, enc3_out], 1))
        up3_out = self.br3(up3_out)
        dec3_out = up3_out

        up4_out = self.up4(torch.cat([dec3_out, enc2_out], 1))
        up4_out = self.br4(up4_out)
        dec4_out = up4_out

        up5_out = self.up5(torch.cat([dec4_out, enc1_out], 1))
        up5_out = self.br5(up5_out)
        dec5_out = up5_out
        
        final_out = self.final(dec5_out)
        return final_out

class AttentionUNet(nn.Module):
    def __init__(self,img_ch=2,output_ch=1):
        super(AttentionUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,size=5,stride=1,padding=1)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256,size=10,stride=1,padding=1)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,size=20,stride=1,padding=1)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,size=41,stride=1,padding=1)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self, Ninput, MR):
        x = torch.cat([Ninput, MR], dim=1)
        
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)     
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1    
