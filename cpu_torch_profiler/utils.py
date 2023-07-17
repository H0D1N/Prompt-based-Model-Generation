import math
import numpy as np
from timm import create_model
from torch import nn
import re
import matplotlib.pyplot as plt
from utils import *
import numpy as np

# generate config:[0.5,0.62,0.4....]  usage of each decision, length=32
def generate_configs(length, start, stop, step):
    # configs contains several config which is a list of 32 random numbers from 0 to 1
    configs = [[x]*length for x in np.arange(start, stop, step)]
    return configs
 

def get_model(config):
    basic_model=create_model('resnet50',features_only=True,out_indices= (2, 3, 4))
    i=0
    for layer in basic_model.layer1:
        purn_channels=math.ceil(layer.conv1.out_channels*config[i])


        new_conv= nn.Conv2d(in_channels=layer.conv1.in_channels,out_channels=purn_channels,stride=layer.conv1.stride[0]
                            ,padding=layer.conv1.padding[0],kernel_size=layer.conv1.kernel_size[0],bias=layer.conv1.bias)
        layer.conv1=new_conv
        layer.bn1=nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i+1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                             stride=layer.conv2.stride[0]
                             , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],bias=layer.conv2.bias)
        layer.conv2= new_conv_2
        layer.bn2=nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3= nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                             stride=layer.conv3.stride[0]
                             , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],bias=layer.conv3.bias)
        layer.conv3=new_conv_3

        i+=2

    for layer in basic_model.layer2:
        purn_channels=math.ceil(layer.conv1.out_channels*config[i])


        new_conv= nn.Conv2d(in_channels=layer.conv1.in_channels,out_channels=purn_channels,stride=layer.conv1.stride[0]
                            ,padding=layer.conv1.padding[0],kernel_size=layer.conv1.kernel_size[0],bias=layer.conv1.bias)
        layer.conv1=new_conv
        layer.bn1=nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i+1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                             stride=layer.conv2.stride[0]
                             , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],bias=layer.conv2.bias)
        layer.conv2= new_conv_2
        layer.bn2=nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3= nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                             stride=layer.conv3.stride[0]
                             , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],bias=layer.conv3.bias)
        layer.conv3=new_conv_3

        i+=2


    for layer in basic_model.layer3:
        purn_channels=math.ceil(layer.conv1.out_channels*config[i])


        new_conv= nn.Conv2d(in_channels=layer.conv1.in_channels,out_channels=purn_channels,stride=layer.conv1.stride[0]
                            ,padding=layer.conv1.padding[0],kernel_size=layer.conv1.kernel_size[0],bias=layer.conv1.bias)
        layer.conv1=new_conv
        layer.bn1=nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i+1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                             stride=layer.conv2.stride[0]
                             , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],bias=layer.conv2.bias)
        layer.conv2= new_conv_2
        layer.bn2=nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3= nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                             stride=layer.conv3.stride[0]
                             , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],bias=layer.conv3.bias)
        layer.conv3=new_conv_3

        i+=2


    for layer in basic_model.layer4:
        purn_channels=math.ceil(layer.conv1.out_channels*config[i])


        new_conv= nn.Conv2d(in_channels=layer.conv1.in_channels,out_channels=purn_channels,stride=layer.conv1.stride[0]
                            ,padding=layer.conv1.padding[0],kernel_size=layer.conv1.kernel_size[0],bias=layer.conv1.bias)
        layer.conv1=new_conv
        layer.bn1=nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i+1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                             stride=layer.conv2.stride[0]
                             , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],bias=layer.conv2.bias)
        layer.conv2= new_conv_2
        layer.bn2=nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3= nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                             stride=layer.conv3.stride[0]
                             , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],bias=layer.conv3.bias)
        layer.conv3=new_conv_3

        i+=2

        return basic_model

def plot_fig(file_name, start, stop ,step):
    with open(file_name + ".txt") as f:
        lines = f.readlines()
        count = 1
        timeings = []
        for line in lines:
            if count % 3 == 0:
                timeings.append(float(re.findall(r"\d+\.?\d*", line)[0]))
            count += 1
    fig, ax = plt.subplots()
    ax.set_xlabel("Config")
    ax.set_ylabel("Time/ms")
    ax.set_title("Model inference time with_" + file_name.split("/")[-1])
    ax.plot(np.arange(start, stop, step), timeings)
    plt.savefig(file_name + ".png")   