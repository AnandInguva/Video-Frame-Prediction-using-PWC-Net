import numpy as np
import torch
from torch import nn
from torchsummary import summary
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
print('--------------------')
print(device)
print('--------------------')
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)   
    def forward(self, x):
        x = self.conv(x)
        return x
class Extractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Extractor, self).__init__()
        self.leakyRelu = nn.LeakyReLU(inplace = False, negative_slope = 0.1)
        self.conv1 = nn(Conv2D(in_channels, 16)
        self.conv2 = Conv2D(16, 16, 1)
        self.conv3 = Conv2D(16, 32)
        self.conv4 = Conv2D(32, 32, 1)
        self.conv5 = Conv2D(32, 64)
        self.conv6 = Conv2D(64, 64, 1)
        self.conv7 = Conv2D(64, 96)
        self.conv8 = Conv2D(96, 96, 1)
        self.conv9 = Conv2D(96, 128)
        self.conv10 = Conv2D(128, 128, 1)
        self.conv11 = Conv2D(128, out_channels)
        self.conv12 = Conv2D(out_channels, out_channels, 1)   
    def forward(self, x):
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.conv2(x))
        c1 = x
        x = self.leakyRelu(self.conv3(x))
        x = self.leakyRelu(self.conv4(x))
        c2 = x
        x = self.leakyRelu(self.conv5(x))
        x = self.leakyRelu(self.conv6(x))
        c3 = x
        x = self.leakyRelu(self.conv7(x))
        x = self.leakyRelu(self.conv8(x))
        c4 = x
        x = self.leakyRelu(self.conv9(x))
        x = self.leakyRelu(self.conv10(x))
        c5 = x
        x = self.leakyRelu(self.conv11(x))
        x = self.leakyRelu(self.conv12(x))
        c6 = x

        return [c1, c2, c3, c4, c5, c6]
                
        
        

model = Extractor(3, 192)
model = model.to(device)
print(model)
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(parameters)
summary(model, (3, 512, 512))