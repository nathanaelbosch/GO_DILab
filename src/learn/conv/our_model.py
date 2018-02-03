"""
Bit more refined towards we actually do and use
"""

try:
    import torch
    from torch import nn
except Exception:
    pass
# import torch.nn.functional as F


class ConvNet(nn.Module):
    """Model strongly inspired from Alphago

    Consists of an initial 5x5 convolution, followed by n 3x3 convolutions,
    with a single 1x1 convolution at the end (while having a different bias
    for each entry)
    """
    def __init__(self, in_channels, conv_depth=9, n_filters=64):
        super(ConvNet, self).__init__()

        # _, in_channels, in1, in2 = input_dim

        self.conv_block = ConvolutionalBlock(in_channels, n_filters)
        residual_blocks = [ResidualBlock(n_filters, n_filters)
                           for _ in range(conv_depth)]
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.policy_head = PolicyHead(n_filters)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        x = self.policy_head(x)
        return x


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, n_filters=256):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=3,
            padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=3,
            padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=3,
            padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual1 = self.conv1(x)
        residual2 = self.batchnorm1(residual1)
        # residual2 = residual1
        residual3 = self.relu1(residual2)
        residual4 = self.conv2(residual3)
        residual5 = self.batchnorm2(residual4)
        x = x + residual5
        x = self.relu2(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=1)
        self.batchnorm = torch.nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2*9*9, 9*9+1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = x.view(-1, 2*9*9)
        x = self.fc(x)
        return x
