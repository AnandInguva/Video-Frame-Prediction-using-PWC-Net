import torch
import torchvision
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):

    '''
    Convolution block of resent: Conv2d -> Batch Norm -> ReLU
    '''

    def __init__(self, in_channels, out_channels, padding = 1, kernel_size = 3, stride = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding = padding, kernel_size = kernel_size,
                                stride = stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Connection(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockUnet(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels = None, up_conv_out_channels = None):
        super().__init__()
        if not up_conv_in_channels:
            up_conv_in_channels = in_channels
        if not up_conv_out_channels:
            up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, 2, 2)
        self.convblock1 = ConvBlock(in_channels, out_channels)
        self.convblock2 = ConvBlock(out_channels, out_channels)

    def forward(self, up, down):
        x = self.upsample(up)
        x = torch.cat([x, down], 1)
        x = self.convblock1(x)
        x = self.convblock2(x)
        return x

class UnetWithResnet(nn.Module):
    DEPTH = 6
    def __init__(self, channels = 3):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained = True)
        upblocks, downblocks = [], []
        self.inputBlock = nn.Sequential()
        resnetLayer = list(resnet.children())[:3]
        for i, layer in enumerate(resnetLayer):
            self.inputBlock.add_module(str(i), layer)
        self.inputPool = list(resnet.children())[3]

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                downblocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(downblocks)
        self.connection = Connection(2048, 2048)
        upblocks.append(UpBlockUnet(2048, 1024))
        upblocks.append(UpBlockUnet(1024, 512))
        upblocks.append(UpBlockUnet(512, 256))
        upblocks.append(UpBlockUnet(192, 128, 256, 128))
        upblocks.append(UpBlockUnet(67, 64, 128, 64))
        self.upblocks = nn.ModuleList(upblocks)
        self.out = nn.Conv2d(64, channels, 1, 1)

    def forward(self, x):
        layers = {}
        layers['layer_0'] = x
        x = self.inputBlock(x)
        layers['layer_1'] = x
        x = self.inputPool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            layers['layer_{}'.format(i)] = x
        x = self.connection(x)
        for i, block in enumerate(self.upblocks, 1):
            key = 'layer_{}'.format(UnetWithResnet.DEPTH - 1 - i)
            x = block(x, layers[key])
        output_feature_map = x
        x = self.out(x)
        return x

model = UnetWithResnet()
def ResNet():
    return model
model = model.cuda()

summary(model, (3,128,128))
#
# model.load_state_dict(torch.load('resnet_weights.pt'))
# print('Weights are loaded')
