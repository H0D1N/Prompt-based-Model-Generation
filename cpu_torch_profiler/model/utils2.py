import math
import torch.nn as nn
import numpy as np
import re
from matplotlib import pyplot as plt
from .efficientdet import EfficientDet
from .config import get_efficientdet_config

# generate config:[0.5,0.62,0.4....]  usage of each decision
def generate_configs(length, start, stop, step):
    # configs contains several config which is a list of 32 random numbers from 0 to 1
    configs = [[x]*length for x in np.arange(start, stop, step)]
    return configs

def get_model(config):
    model_config = get_efficientdet_config('resdet50')
    model =EfficientDet(config=model_config,pretrained_backbone=False)
    i=0
    for layer in model.backbone.layer1:
        purn_channels = math.ceil(layer.conv1.out_channels * config[i])

        new_conv = nn.Conv2d(in_channels=layer.conv1.in_channels, out_channels=purn_channels, stride=layer.conv1.stride[0]
                            , padding=layer.conv1.padding[0], kernel_size=layer.conv1.kernel_size[0],
                            bias=layer.conv1.bias)
        layer.conv1 = new_conv
        layer.bn1 = nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i + 1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                            stride=layer.conv2.stride[0]
                            , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],
                            bias=layer.conv2.bias)
        layer.conv2 = new_conv_2
        layer.bn2 = nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3 = nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                            stride=layer.conv3.stride[0]
                            , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],
                            bias=layer.conv3.bias)
        layer.conv3 = new_conv_3

        i += 2
    for layer in model.backbone.layer2:
        purn_channels = math.ceil(layer.conv1.out_channels * config[i])

        new_conv = nn.Conv2d(in_channels=layer.conv1.in_channels, out_channels=purn_channels, stride=layer.conv1.stride[0]
                            , padding=layer.conv1.padding[0], kernel_size=layer.conv1.kernel_size[0],
                            bias=layer.conv1.bias)
        layer.conv1 = new_conv
        layer.bn1 = nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i + 1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                            stride=layer.conv2.stride[0]
                            , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],
                            bias=layer.conv2.bias)
        layer.conv2 = new_conv_2
        layer.bn2 = nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3 = nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                            stride=layer.conv3.stride[0]
                            , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],
                            bias=layer.conv3.bias)
        layer.conv3 = new_conv_3

        i += 2
    for layer in model.backbone.layer3:
        purn_channels = math.ceil(layer.conv1.out_channels * config[i])

        new_conv = nn.Conv2d(in_channels=layer.conv1.in_channels, out_channels=purn_channels, stride=layer.conv1.stride[0]
                            , padding=layer.conv1.padding[0], kernel_size=layer.conv1.kernel_size[0],
                            bias=layer.conv1.bias)
        layer.conv1 = new_conv
        layer.bn1 = nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i + 1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                            stride=layer.conv2.stride[0]
                            , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],
                            bias=layer.conv2.bias)
        layer.conv2 = new_conv_2
        layer.bn2 = nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3 = nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                            stride=layer.conv3.stride[0]
                            , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],
                            bias=layer.conv3.bias)
        layer.conv3 = new_conv_3

        i += 2
    for layer in model.backbone.layer4:
        purn_channels = math.ceil(layer.conv1.out_channels * config[i])

        new_conv = nn.Conv2d(in_channels=layer.conv1.in_channels, out_channels=purn_channels, stride=layer.conv1.stride[0]
                            , padding=layer.conv1.padding[0], kernel_size=layer.conv1.kernel_size[0],
                            bias=layer.conv1.bias)
        layer.conv1 = new_conv
        layer.bn1 = nn.BatchNorm2d(purn_channels)

        purn_channels_2 = math.ceil(layer.conv2.out_channels * config[i + 1])
        new_conv_2 = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                            stride=layer.conv2.stride[0]
                            , padding=layer.conv2.padding[0], kernel_size=layer.conv2.kernel_size[0],
                            bias=layer.conv2.bias)
        layer.conv2 = new_conv_2
        layer.bn2 = nn.BatchNorm2d(num_features=purn_channels_2)

        new_conv_3 = nn.Conv2d(in_channels=purn_channels_2, out_channels=layer.conv3.out_channels,
                            stride=layer.conv3.stride[0]
                            , padding=layer.conv3.padding[0], kernel_size=layer.conv3.kernel_size[0],
                            bias=layer.conv3.bias)
        layer.conv3 = new_conv_3

        i += 2
    #class_net
    purn_channels = math.ceil(model.class_net.conv_rep[0].conv.out_channels * config[i])

    new_conv = nn.Conv2d(in_channels=model.class_net.conv_rep[0].conv.in_channels, out_channels=purn_channels, stride=model.class_net.conv_rep[0].conv.stride[0]
                        , padding=model.class_net.conv_rep[0].conv.padding[0], kernel_size=model.class_net.conv_rep[0].conv.kernel_size[0],
                        bias=model.class_net.conv_rep[0].conv.bias)
    model.class_net.conv_rep[0].conv = new_conv
    for n in range(model_config.num_levels):
        model.class_net.bn_rep[0][n] = nn.BatchNorm2d(purn_channels)

    i=i+1

    purn_channels_2 = math.ceil(model.class_net.conv_rep[1].conv.out_channels * config[i])

    new_conv = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                        stride=model.class_net.conv_rep[1].conv.stride[0]
                        , padding=model.class_net.conv_rep[1].conv.padding[0],
                        kernel_size=model.class_net.conv_rep[1].conv.kernel_size[0],
                        bias=model.class_net.conv_rep[1].conv.bias)
    model.class_net.conv_rep[1].conv = new_conv
    for n in range(model_config.num_levels):
        model.class_net.bn_rep[1][n] = nn.BatchNorm2d(purn_channels_2)
    i=i+1

    purn_channels_3 = math.ceil(model.class_net.conv_rep[2].conv.out_channels * config[i])

    new_conv = nn.Conv2d(in_channels=purn_channels_2, out_channels=purn_channels_3,
                        stride=model.class_net.conv_rep[2].conv.stride[0]
                        , padding=model.class_net.conv_rep[2].conv.padding[0],
                        kernel_size=model.class_net.conv_rep[2].conv.kernel_size[0],
                        bias=model.class_net.conv_rep[2].conv.bias)
    model.class_net.conv_rep[2].conv = new_conv
    for n in range(model_config.num_levels):
        model.class_net.bn_rep[2][n] = nn.BatchNorm2d(purn_channels_3)

    new_conv=nn.Conv2d(in_channels=purn_channels_3,out_channels=model.class_net.predict.conv.out_channels,
                    stride=model.class_net.predict.conv.stride[0],kernel_size=model.class_net.predict.conv.kernel_size[0],
                    padding=model.class_net.predict.conv.padding[0])

    bias=model.class_net.predict.conv.bias.data
    new_conv.bias.data.copy_(bias)
    model.class_net.predict.conv=new_conv
    i=i+1

    #box_net

    purn_channels = math.ceil(model.box_net.conv_rep[0].conv.out_channels * config[i])

    new_conv = nn.Conv2d(in_channels=model.box_net.conv_rep[0].conv.in_channels, out_channels=purn_channels,
                        stride=model.box_net.conv_rep[0].conv.stride[0]
                        , padding=model.box_net.conv_rep[0].conv.padding[0],
                        kernel_size=model.box_net.conv_rep[0].conv.kernel_size[0],
                        bias=model.box_net.conv_rep[0].conv.bias)
    model.box_net.conv_rep[0].conv = new_conv

    for n in range(model_config.num_levels):
        model.box_net.bn_rep[0][n] = nn.BatchNorm2d(purn_channels)

    i = i + 1

    purn_channels_2 = math.ceil(model.box_net.conv_rep[1].conv.out_channels * config[i])

    new_conv = nn.Conv2d(in_channels=purn_channels, out_channels=purn_channels_2,
                        stride=model.box_net.conv_rep[1].conv.stride[0]
                        , padding=model.box_net.conv_rep[1].conv.padding[0],
                        kernel_size=model.box_net.conv_rep[1].conv.kernel_size[0],
                        bias=model.box_net.conv_rep[1].conv.bias)
    model.box_net.conv_rep[1].conv = new_conv

    for n in range(model_config.num_levels):
        model.box_net.bn_rep[1][n] = nn.BatchNorm2d(purn_channels_2)
    i = i + 1

    purn_channels_3 = math.ceil(model.box_net.conv_rep[2].conv.out_channels * config[i])

    new_conv = nn.Conv2d(in_channels=purn_channels_2, out_channels=purn_channels_3,
                        stride=model.box_net.conv_rep[2].conv.stride[0]
                        , padding=model.box_net.conv_rep[2].conv.padding[0],
                        kernel_size=model.box_net.conv_rep[2].conv.kernel_size[0],
                        bias=model.box_net.conv_rep[2].conv.bias)
    model.box_net.conv_rep[2].conv = new_conv

    for n in range(model_config.num_levels):
        model.box_net.bn_rep[2][n] = nn.BatchNorm2d(purn_channels_3)

    new_conv = nn.Conv2d(in_channels=purn_channels_3, out_channels=model.box_net.predict.conv.out_channels,
                        stride=model.box_net.predict.conv.stride[0],
                        kernel_size=model.box_net.predict.conv.kernel_size[0],
                        padding=model.box_net.predict.conv.padding[0])

    bias=model.box_net.predict.conv.bias.data
    new_conv.bias.data.copy_(bias)
    model.box_net.predict.conv = new_conv

    i = i + 1
    return model.backbone, model.fpn, model.class_net, model.box_net, model
