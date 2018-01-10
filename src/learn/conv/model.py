import torch


class ConvNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, conv_depth=5):
        super(ConvNet, self).__init__()

        _, in_channels, in1, in2 = input_dim
        n_filters = 128

        self.start_conv = torch.nn.Conv2d(
            in_channels,
            out_channels=n_filters,
            kernel_size=5,
            padding=2)
        self.start_relu = torch.nn.ReLU()

        self.mid_convs = []
        self.relus = []
        for i in range(conv_depth):
            self.mid_convs.append(torch.nn.Conv2d(
                n_filters,
                out_channels=n_filters,
                kernel_size=3,
                padding=1))
            self.relus.append(torch.nn.ReLU())

        self.last_conv = torch.nn.Conv2d(
            n_filters,
            out_channels=1,
            kernel_size=1)

        self.end_relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(128*9*9, 9*9+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.start_conv(x)
        x = self.start_relu(x)
        for conv, relu in zip(self.mid_convs, self.relus):
            x = conv(x)
            x = relu(x)
        # x = self.last_conv(x)
        # x = self.softmax(x)
        x = x.view(-1, 128*9*9)

        x = self.fc(x)
        x = self.softmax(x)
        return x

    def cuda(self):
        super(ConvNet, self).cuda()
        for conv in self.mid_convs:
            conv.cuda()


