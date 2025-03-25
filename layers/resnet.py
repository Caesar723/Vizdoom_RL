import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1#resnet18
    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class ResNet(nn.Module):
    def __init__(self, layers, channel_in=3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False)  # (N, 3, H, W) -> (N, 64, H/2, W/2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (N, 64, H/2, W/2) -> (N, 64, H/4, W/4)

        self.layer1 = self._make_layer( 64,  layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        #self.layer3 = self._make_layer(256, layers[2], stride=2)
        #self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        
        # 如果输入通道数 != 输出通道数，或者 stride != 1，需要下采样
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):#(N, 3, H, W)
        x = self.conv1(x)#(N, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#(N, 64, H/4, W/4)
        x = self.layer1(x)#(N, 64, H/4, W/4)
        x = self.layer2(x)#(N, 128, H/8, W/8)
        #x = self.layer3(x)#(N, 256, H/16, W/16)
        #x = self.layer4(x)#(N, 512, H/32, W/32)
        x = self.avgpool(x)#(N, 512, 1, 1)
        x = x.view(x.size(0), -1)#(N, 512)
        return x
    

if __name__ == "__main__":
    resnet = ResNet([2, 2, 2, 2], channel_in=1)
    x = torch.randn(3,1, 1, 224, 224)
    x=x.view(3*1,1,224,224)
    x=resnet(x)
    x=x.view(3,1,-1)
    print(x.shape)

