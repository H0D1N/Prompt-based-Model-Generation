import math
import torch.nn as nn
import numpy as np
import json
import random
from matplotlib import pyplot as plt
from .efficientdet import EfficientDet
from .config import get_efficientdet_config

def configs_random(configs_num):
    config_indices = []
    for num in range(1, 19):
        if num < 1 or num > 18:
            raise ValueError("Invalid num. num should be an integer between 1 and 18.")
        elif num >= 1 and num <= 16:
            config_indices.append([2 * num - 2, 2 * num - 1])
        elif num == 17:
            config_indices.append([32, 33, 34])
        elif num == 18:
            config_indices.append([35, 36, 37])
    configs = [[round(random.uniform(0.05, 1), 4) for _ in range(38)] for a in range(configs_num)]
    return config_indices, configs

# 针对第module_num个module，生成所有可能的config，无关位置的值为1
def configs_generate(module_num):
    if module_num < 1 or module_num > 18:
        raise ValueError("Invalid module_num. module_num should be an integer between 1 and 18.")

    if module_num >= 1 and module_num <= 16:
        num_combinations = 10 ** 2
    else:
        num_combinations = 10 ** 3

    # Initialize the configs matrix with all 1s
    configs = [[1] * 38 for _ in range(num_combinations)]

    # 生成从0.1到1，步长为0.1的列表
    value_range = [round(0.1 + 0.1 * i, 1) for i in range(10)]

    # Generate the config indices based on module_num
    if module_num >= 1 and module_num <= 16:
        config_indices = [2 * module_num - 2, 2 * module_num - 1]

        for i in range(num_combinations):
            configs[i][config_indices[0]] = value_range[i % 10]
            configs[i][config_indices[1]] = value_range[i // 10]

    elif module_num == 17:
        config_indices = [32, 33, 34]

        for i in range(num_combinations):
            configs[i][config_indices[0]] = value_range[i % 10]
            configs[i][config_indices[1]] = value_range[(i // 10) % 10 ]
            configs[i][config_indices[2]] = value_range[i // 100]

    elif module_num == 18:
        config_indices = [35, 36, 37]

        for i in range(num_combinations):
            configs[i][config_indices[0]] = value_range[i % 10]
            configs[i][config_indices[1]] = value_range[(i // 10) % 10  ]
            configs[i][config_indices[2]] = value_range[i // 100]

    return config_indices, configs

# 针对生成的trace.json文件，提取对应module的延时
def get_module_latency(trace_name, runs):
    modules = 18
    ts_difference_matrix = []
    with open(trace_name, 'r') as file:
        raw_data = json.load(file)

        # 不同次run的结果的ProfilerStep位置
        ProfilerStep_pos = list(map(lambda str: "ProfilerStep" in str["name"], raw_data))
        ProfilerStep_pos = [index for index, value in enumerate(ProfilerStep_pos) if value]
        # print(ProfilerStep_pos)

        # 不同次run的结果的Conv2d位置
        Conv2dStep_pos = list(map(lambda str: "aten::conv2d" in str["name"], raw_data))
        Conv2dStep_pos = [index for index, value in enumerate(Conv2dStep_pos) if value]
        # print(Conv2dStep_pos)

        # 不同次run的结果的BN位置
        BNStep_pos = list(map(lambda str: "aten::batch_norm" in str["name"], raw_data))
        BNStep_pos = [index for index, value in enumerate(BNStep_pos) if value]
        
        # 不同次run的结果的ReLu位置
        ReLuStep_pos = list(map(lambda str: "aten::relu_" in str["name"], raw_data))
        ReLuStep_pos = [index for index, value in enumerate(ReLuStep_pos) if value]


        def conv2d_timestamp_difference(index2, index1):
            ts_difference = raw_data[Conv2dStep_pos[index2]]['ts'] - raw_data[Conv2dStep_pos[index1]]['ts']
            return ts_difference


        ts_difference_list = [0] * (modules * runs)  # 初始化一个包含runs个周期的总列表，每个周期有modules个元素，初始值为0

        for run in range(runs):  # 进行runs次循环

            period_index = run * 18
            conv2d_period_index = run * 131
            # ###############################Overhead###############################
            # 最顶层有一个conve2d

            # ###############################layer1#################################
          
            ts_difference_list[period_index] = conv2d_timestamp_difference(conv2d_period_index + 5,
                                                                           conv2d_period_index + 1)
            ts_difference_list[period_index + 1] = conv2d_timestamp_difference(conv2d_period_index + 8,
                                                                               conv2d_period_index + 5)
            ts_difference_list[period_index + 2] = conv2d_timestamp_difference(conv2d_period_index + 11,
                                                                               conv2d_period_index + 8)

            ###############################layer2#################################
            ts_difference_list[period_index + 3] = conv2d_timestamp_difference(conv2d_period_index + 15,
                                                                               conv2d_period_index + 11)
            ts_difference_list[period_index + 4] = conv2d_timestamp_difference(conv2d_period_index + 18,
                                                                               conv2d_period_index + 15)
            ts_difference_list[period_index + 5] = conv2d_timestamp_difference(conv2d_period_index + 21,
                                                                               conv2d_period_index + 18)
            ts_difference_list[period_index + 6] = conv2d_timestamp_difference(conv2d_period_index + 24,
                                                                               conv2d_period_index + 18)

            ###############################layer3#################################

            ts_difference_list[period_index + 7] = conv2d_timestamp_difference(conv2d_period_index + 28,
                                                                               conv2d_period_index + 24)
            ts_difference_list[period_index + 8] = conv2d_timestamp_difference(conv2d_period_index + 31,
                                                                               conv2d_period_index + 28)
            ts_difference_list[period_index + 9] = conv2d_timestamp_difference(conv2d_period_index + 34,
                                                                               conv2d_period_index + 31)
            ts_difference_list[period_index + 10] = conv2d_timestamp_difference(conv2d_period_index + 37,
                                                                                conv2d_period_index + 34)
            ts_difference_list[period_index + 11] = conv2d_timestamp_difference(conv2d_period_index + 40,
                                                                                conv2d_period_index + 37)
            ts_difference_list[period_index + 12] = conv2d_timestamp_difference(conv2d_period_index + 43,
                                                                                conv2d_period_index + 40)

            ###############################layer4#################################

            ts_difference_list[period_index + 13] = conv2d_timestamp_difference(conv2d_period_index + 47,
                                                                                conv2d_period_index + 43)
            ts_difference_list[period_index + 14] = conv2d_timestamp_difference(conv2d_period_index + 50,
                                                                                conv2d_period_index + 47)
            ts_difference_list[period_index + 15] = conv2d_timestamp_difference(conv2d_period_index + 53,
                                                                                conv2d_period_index + 50)

            ###############################(class_net): HeadNet#################################
            ts_difference_list[period_index + 16] = conv2d_timestamp_difference(conv2d_period_index + 131 - 20,
                                                                                conv2d_period_index + 131 - 40)

            ###############################(box_net): HeadNet#################################
            # 此处理想是131, 但是最后一次会超出index,只能改为130
            ts_difference_list[period_index + 17] = conv2d_timestamp_difference(conv2d_period_index + 130,  
                                                                                conv2d_period_index + 131 - 20)
        # 转为modules * runs的矩阵
        ts_difference_matrix = np.transpose(np.reshape(ts_difference_list, (runs, modules)))

        # 模型的总延时, 长度runs
        total_Profiler_latency = [raw_data[pos]['dur'] for pos in ProfilerStep_pos]

    return ts_difference_matrix, total_Profiler_latency

# generate config:[0.5,0.62,0.4....]  usage of each decision
def generate_configs(length, start, stop, step):
    # configs contains several config which is a list of 32 random numbers from 0 to 1
    configs = [[x]*length for x in np.arange(start, stop, step)]
    return configs

# generate model according to config
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
    return model

