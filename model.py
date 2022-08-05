import torch
from torch import nn

class IBN(nn.Module):
    def __init__(self, in_channel: int, type: str = 'a'):
        super(IBN, self).__init__()
        self.type = type
        self.in_channel = in_channel
        self.half = int(in_channel/2)
        self.half_IN = nn.InstanceNorm2d(self.half, affine=True)
        self.half_BN = nn.BatchNorm2d(self.in_channel - self.half)
        self.IN = nn.InstanceNorm2d(self.in_channel, affine=True)
        self.BN = nn.BatchNorm2d(self.in_channel)
        
    def forward(self, input):
        if self.type == 'a':
            split = torch.split(input, self.half, 1)
            out1 = self.half_IN(split[0].contiguous())
            out2 = self.half_BN(split[1].contiguous())
            return torch.cat((out1, out2), 1)
        if self.type == 'b':
            return self.IN(input)
        if self.type == 'c':
            in1 = input.contiguous()
            out1 = self.IN(input)
            out2 = self.BN(in1)
            return out1 + out2
        if self.type == 'd':
            split = torch.split(input, self.half, 1)
            out1 = self.IN(split[0].contiguous())
            out2 = split[1].contiguous()
            return torch.cat((out1, out2), 1)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel: int, out_channel: int, ibn: bool = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(in_channel, type='a')
        else:
            self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

if __name__ == "__main__":
    input = torch.randn(1, 3, 112, 112)
    b = Bottleneck(3, 3, True)
    out = b.forward(input)