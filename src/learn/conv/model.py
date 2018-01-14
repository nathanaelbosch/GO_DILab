import torch
from torch import nn
# import torch.nn.functional as F


class ConvNet(nn.Module):
    """Model strongly inspired from Alphago

    Consists of an initial 5x5 convolution, followed by n 3x3 convolutions,
    with a single 1x1 convolution at the end (while having a different bias
    for each entry)
    """
    def __init__(self, input_dim, output_dim, conv_depth=5):
        super(ConvNet, self).__init__()

        _, in_channels, in1, in2 = input_dim
        n_filters = 128

        # First 5x5 Convolution + Relu
        self.start_conv = nn.Conv2d(
            in_channels,
            out_channels=n_filters,
            kernel_size=5,
            padding=2)
        self.start_relu = nn.ReLU()

        # `conv_depth` many 3x3 convolutions + Relu
        layers = []
        for i in range(conv_depth):
            layers.append(nn.Conv2d(
                n_filters,
                out_channels=n_filters,
                kernel_size=3,
                padding=1))
            layers.append(nn.ReLU())
        self.mid_convs = nn.Sequential(*layers)

        # Last 1x1 conv: Basically a single weight and 81 biases
        self.last_conv = Conv1x1((n_filters, in1, in2))

        # self.last_conv = nn.Conv2d(
        #     n_filters,
        #     out_channels=1,
        #     kernel_size=1)

        # self.end_relu = nn.ReLU()
        # self.fc = nn.Linear(128*9*9, 9*9+1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.start_conv(x)
        x = self.start_relu(x)
        x = self.mid_convs(x)
        x = self.last_conv(x)
        # x = self.softmax(x)

        # x = self.fc(x)
        x = self.softmax(x)
        x = x.view(-1, 128*9*9)
        return x


class Conv1x1(nn.Module):
    def __init__(self, shape, init=0.0001):
        super(Conv1x1, self).__init__()
        self.weight = nn.Parameter(torch.ones(1)*init, requires_grad=True)
        self.biases = nn.Parameter(torch.ones(shape)+init, requires_grad=True)

    def forward(self, x):
        # print(self.weight)
        return x * self.weight + self.biases
