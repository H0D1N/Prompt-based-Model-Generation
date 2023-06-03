import torch.nn as nn


class SelectedResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        # Conv1
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # Conv2
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

        # act

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, filter_select=None):
        if filter_select != None:
            # filter_select: [batach_size,conv_num,out_channels]
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)
            out = out * filter_select[:, 0, :][:, :, None, None]

            out = self.conv2(out)

            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.act(out)
            out = out * filter_select[:, 1, :][:, :, None, None]

            return out

        else:
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.act(out)
            return out

            return x