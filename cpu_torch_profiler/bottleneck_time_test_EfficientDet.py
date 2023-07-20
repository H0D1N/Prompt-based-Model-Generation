import json
import numpy as np

file_prefix = "data/batch_2/config_"  # 文件名前缀
file_suffix = "_profiler_trace.json"  # 文件名后缀
num_files = 10  # 文件数量
run_period = 4
statistics_num = 18
''' 
Conv2d在模型中的分布

 Overhead,1个Conv2d
    layer1中
        (0): Bottleneck 4个Conv2d
        (1): Bottleneck 3个Conv2d
        (2): Bottleneck 3个Conv2d
    layer2中
        (0): Bottleneck 4个Conv2d
        (1): Bottleneck 3个Conv2d
        (2): Bottleneck 3个Conv2d  
        (3): Bottleneck 3个Conv2d  
     layer3中
        (0): Bottleneck 4个Conv2d
        (1): Bottleneck 3个Conv2d
        (2): Bottleneck 3个Conv2d
        (3): Bottleneck 3个Conv2d
        (4): Bottleneck 3个Conv2d
        (5): Bottleneck 3个Conv2d
     layer4中
        (0): Bottleneck 4个Conv2d
        (1): Bottleneck 3个Conv2d
        (2): Bottleneck 3个Conv2d     
     共53个    
         
    fpn: 38个
    
    (class_net): HeadNet 20个
    (box_net): HeadNet 20个

ProfilerStep_pos = [0, 4709, 9418, 14127]
Conv2dStep_pos = [1, 28, 54, 79, 103, 130, 156, 181, 208, 234, 259, 286, 312, 337, 361, 387, 413, 438, 465, 491, 516, 543, 569, 594,
      621, 647, 672, 696, 722, 748, 773, 800, 826, 851, 878, 904, 929, 956, 982, 1007, 1034, 1060, 1085, 1112, 1138,
      1163, 1187, 1213, 1239, 1264, 1291, 1317, 1342, 1369, 1446, 1469, 1545, 1568, 1644, 1667, 1741, 1764, 1840, 1863,
      1939, 2011, 2072, 2144, 2219, 2294, 2367, 2442, 2517, 2589, 2650, 2722, 2797, 2872, 2945, 3020, 3095, 3167, 3228,
      3300, 3375, 3450, 3523, 3598, 3673, 3745, 3806, 3829, 3854, 3879, 3904, 3917, 3942, 3967, 3992, 4005, 4030, 4055,
      4080, 4093, 4118, 4143, 4168, 4181, 4206, 4231, 4256, 4269, 4294, 4319, 4344, 4357, 4382, 4407, 4432, 4445, 4470,
      4495, 4520, 4533, 4558, 4583, 4608, 4621, 4646, 4671, 4696]

print(len(Conv2dStep_pos))
run一次, 有131个Conv2d
'''

overhead_latency_matrix = np.zeros((num_files, run_period))  # 初始化一个10行, 4列的矩阵
for i in range(num_files):
    file_name = file_prefix + str((i + 1) / 10) + file_suffix  # 构建文件名
    print("Reading file:", file_name)

    with open(file_name, 'r') as file:
        raw_data = json.load(file)["traceEvents"]

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


        ts_difference_list = [0] * (
                statistics_num * run_period)  # 初始化一个包含run_period个周期的总列表，每个周期有statistics_num个元素，初始值为0

        for run in range(4):  # 进行4次循环

            period_index = run * 18
            conv2d_period_index = run * 131
            #
            # ###############################Overhead###############################

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
            # ts_difference_list[period_index + 17] = raw_data[Conv2dStep_pos[conv2d_period_index + 130] + 12]['ts'] - raw_data[Conv2dStep_pos[conv2d_period_index + 131 - 20]]['ts']
            ts_difference_list[period_index + 17] = conv2d_timestamp_difference(conv2d_period_index + 130,  # 此处理想是131
                                                                                # ，但是最后一次会超出index,只能改为130
                                                                                conv2d_period_index + 131 - 20)


        # 转为4*18的矩阵
        ts_difference_matrix = np.reshape(ts_difference_list, (4, 18))

        # 对每行求和，计算18个module的延时之和
        row_sums = np.sum(ts_difference_matrix, axis=1)
        bottleneck_latency_sum = row_sums.tolist()
        print("-- 18 modules in 4 runs: \n", bottleneck_latency_sum, "   Average:", np.mean(bottleneck_latency_sum))

        # 模型的总延时, 长度4，代表4个run
        total_Profiler_latency = [raw_data[pos]['dur'] for pos in ProfilerStep_pos]
        print("-- model inference in 4 runs: \n", total_Profiler_latency, "  Average:", np.mean(total_Profiler_latency))

        # 计算18个module延时之和在总延时中的占比
        bottleneck_latency_percentage = [round(bottleneck_latency_sum[i] / total_Profiler_latency[i], 3) for i in
                                            range(len(total_Profiler_latency))]
        print("-- percentage of 18 modules in 4 runs: \n", bottleneck_latency_percentage, "  Average:", np.mean(bottleneck_latency_percentage))


        # 两个list做差：total_Profiler_latency[i] - bottleneck_latency_sum[i]，得到Overhead_latency
        Overhead_latency = [total_Profiler_latency[i] - bottleneck_latency_sum[i] for i in
                            range(len(total_Profiler_latency))]
        print("-- Overhead in 4 runs: \n", Overhead_latency, "  Average: ", np.mean(Overhead_latency))
        print("percentage of Overhead in 4 runs: \n", [round(Overhead_latency[i] / total_Profiler_latency[i], 3) for i in
                            range(len(total_Profiler_latency))], "  Average:", round(np.mean(Overhead_latency) / np.mean(total_Profiler_latency), 3), "\n")
        
        overhead_latency_matrix[i, :] = Overhead_latency  # 填入10*4的Overhead_latency矩阵中