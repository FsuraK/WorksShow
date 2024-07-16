import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_pytorch import SwinTransformer
from torch.nn import Module, Sequential
import torchvision


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Handle case where input and output channels are different
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              padding=0) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.skip:
            residual = self.skip(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualBlockComplex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_prob=0.3):
        super(ResidualBlockComplex, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Handle case where input and output channels are different
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              padding=0) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.skip:
            residual = self.skip(x)

        out += residual
        out = self.relu(out)

        return out


class RainfallEstimationCNN(nn.Module):
    def __init__(self):
        super(RainfallEstimationCNN, self).__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(2, 32),  # 256x256
            nn.MaxPool2d(2),  # 128x128
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),  # 64x64
            ResidualBlock(64, 128),
            nn.MaxPool2d(2)  # 32x32
        )

        self.middle = ResidualBlock(128, 128)  # 32x32

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 64x64
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 128x128
            ResidualBlock(32, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 256x256
            ResidualBlock(16, 16)
        )

        self.output_conv = nn.Conv2d(16, 1, kernel_size=1)  # 1 channel for rainfall

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.output_conv(x)
        return x


class ConvForSizeReduceDetached(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(ConvForSizeReduceDetached, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=times, stride=times, padding=0, bias=False)
        self.times = times

    def forward(self, x):
        # check channels
        if x.size(1) != self.conv.in_channels:
            raise ValueError('Input data channels is not equal model require')

        # check size
        if x.size(-1) < self.times:
            raise ValueError('Input data size is too little')

        return self.conv(x)


class DownSampleConvInput5Dim2(nn.Module):
    """
    :param: 5 dim Input data with shape torch.Tensor([batch, seq_len, channel, weight, height])
    :function: Reduce the dimensions of [weight] and [height] to 1 / times of the original Input
    :return: 5 dim Input data with shape torch.Tensor([batch, seq_len, channel, weight / times, height / times])
    """
    def __init__(self, in_channels, out_channels, times):
        super(DownSampleConvInput5Dim2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.times = times
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.times, padding=1)

    def forward(self, x):
        batch, seq_len, channel, w, h = x.shape
        if channel is not self.in_channels:
            raise ValueError('Input data channels is not equal model require')

        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # Reshape to (batch*seq_len, channel, w, h)
        x = self.conv(x)  # Convolve
        _, _, new_w, new_h = x.shape
        x = x.view(batch, seq_len, -1, new_w, new_h)  # Reshape to the original
        return x


class DownSampleConvInput4Dim2(nn.Module):
    """
    :param: 4 dim Input data with shape torch.Tensor([batch, seq_len, channel, weight, height])
    :function: Reduce the dimensions of [weight] and [height] to 1 / times of the original Input
    :return: 4 dim Input data with shape torch.Tensor([batch, seq_len, channel, weight / times, height / times])
    """
    def __init__(self, seq_len, out_channels, times):
        super(DownSampleConvInput4Dim2, self).__init__()
        self.in_channels = seq_len
        self.out_channels = out_channels
        self.times = times
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.times, padding=1)

    def forward(self, x):
        batch, seq_len, w, h = x.shape
        if seq_len is not self.in_channels:
            raise ValueError('Input data channels is not equal model require')

        x = self.conv(x)
        return x


class Conv2OneChannel(nn.Module):
    """
    :param: 4 dim Input data with shape torch.Tensor([batch, seq_len, channel, weight, height])
    :function: Reduce the dimensions of [weight] and [height] to 1 / times of the original Input
    :return: 4 dim Input data with shape torch.Tensor([batch, seq_len, channel, weight / times, height / times])
    """
    def __init__(self, in_channels, out_channels, times):
        super(Conv2OneChannel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.times = times
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.times, padding=1)

    def forward(self, x):
        _, in_channels, _, _ = x.shape
        if in_channels is not self.in_channels:
            raise ValueError('Input data channels is not equal model require')

        x = self.conv(x)
        return x


class AttentionModule(nn.Module):
    """
    pure Attention
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)
        return out * x


class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SwinTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = identity + x
        identity = x
        x = self.norm2(x)
        x = F.relu(self.mlp(x))
        x = identity + x
        return x


class FeatureFusion(object):
    def __init__(self):
        self.cat_dim = 1

    def StageOneDim4Cat(self, x, y):
        """
        the input x, y should own the same shape, with a dim = 4
        :return: Fusion, cat operation in seq_len
        """
        return torch.cat((x, y), dim=self.cat_dim)

    def StageOneDim4Sum(self, x, y):
        """
        the input x, y should own the same shape, with a dim = 4
        :return: Fusion, sum x and y
        """
        return x + y

    def StageOneDim4Mean(self, x, y):
        """
        the input x, y should own the same shape, with a dim = 4
        :return: Fusion, mean x and y
        """
        return (x + y) / 2

    def StageTwoDim4Cat(self, x, y):
        """
        the input x, y should own the same shape, with a dim = 4
        :param: x -> [batch, 1, w, h]
        :param: y -> [batch, 1, w, h]
        :return: Fusion, cat operation, shape -> [batch, w*h*2]
        """
        batch, _, _, _ = x.shape
        z = torch.cat([x, y], dim=1)
        z = z.view(batch, -1)
        return z
