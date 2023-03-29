import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.LeakyReLU(0.2,inplace=True)
    )


def double_conv(in_c,out_c):
  conv = nn.Sequential(
      nn.Conv2d(in_c,out_c,kernel_size=3),
      nn.BatchNorm2d(out_c),  
      nn.LeakyReLU(0.2,inplace=True),
      nn.Conv2d(out_c,out_c,kernel_size=3),
      nn.BatchNorm2d(out_c),  
      nn.LeakyReLU(0.2,inplace=True),
  )
  return conv




class Resnet_UNet(nn.Module):
    def __init__(self, out_channels=1):

        super().__init__()

        self.reflect2d = nn.ReflectionPad2d(4)
        self.encoder = models.resnet18()


        self.conv1=list(self.encoder._modules.items())[0][1]
        self.bn1=list(self.encoder._modules.items())[1][1]
        self.relu=list(self.encoder._modules.items())[2][1]
        self.maxpool=list(self.encoder._modules.items())[3][1]

        self.block1=list(self.encoder._modules.items())[4][1]
        self.block2=list(self.encoder._modules.items())[5][1]
        self.block3=list(self.encoder._modules.items())[6][1]
        self.block4=list(self.encoder._modules.items())[7][1]

        

        self.up_conv6 = up_conv(512,512)
        self.conv6 = double_conv(512+256, 256)

        self.up_conv7 = up_conv(256, 256)
        self.conv7 = double_conv(256 + 128, 128)

        self.up_conv8 = up_conv(128, 128)
        self.conv8 = double_conv(128+64 , 64)

        self.up_conv9 = up_conv(64, 64)
        self.conv9_end= nn.Conv2d(64, out_channels, kernel_size=1)

        self.conv9 = double_conv(64 , 32)
        
        self.up_conv10 = up_conv(32, 16)

        self.conv10 = nn.Conv2d(16, out_channels, kernel_size=1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x= self.maxpool(x)

        block1 = self.block1(x)                        
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)



        x = self.up_conv6(block4)
        x = torch.cat([x, block3], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = self.reflect2d(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv7(x)


        x = self.up_conv8(x)
        x = self.reflect2d(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = self.reflect2d(x)
        x = self.conv9_end(x)


        return x


