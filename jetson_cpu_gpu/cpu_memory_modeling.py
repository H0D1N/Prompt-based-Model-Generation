import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

runs = 500
ratio = 0.8  # 80% of the data is used for training, and 20% is used for testing
slice_num = int(runs * ratio)  # 800

# 创建一个空列表来存储Self CPU Mem值
Self_CPU_Mem_values = []

# 读取1000个txt文件
for i in range(500):
    filename = f"data/cpu_memory/batch_1/raw_data/run_No.{i}.txt"
    try:
        with open(filename, "r") as file:
            # 逐行读取文件内容
            for line in file:
                # 使用正则表达式匹配包含"ProfilerStep*"的行
                if "ProfilerStep*" in line:
                    # 使用空格分割行的内容
                    values = line.split()
                    if len(values) >= 2:  # 按照大于等于2个空格划分
                        try:
                            Self_CPU_Mem = abs(float(values[8]))
                            Self_CPU_Mem_values.append(Self_CPU_Mem)
                        except ValueError:
                            print(f"文件 {filename} 中的Self CPU Mem值无效。")
                    else:
                        print(f"文件 {filename} 中的行格式不正确。")
                    break  # 停止查找其他行，因为我们只关心第一个匹配项

    except FileNotFoundError:
        # 如果文件不存在，可以做一些处理或记录
        print(f"文件 {filename} 未找到。")

# 打印Self CPU Mem值列表
print('Self CPU Mem值列表:')
print(Self_CPU_Mem_values)
print()

################################### memory predictor using all 38 configs #################################
configs_train = np.ones((slice_num, 39))
configs_valid = np.ones((runs - slice_num, 39))

model_memory_train = np.array(Self_CPU_Mem_values[:slice_num]).reshape(slice_num, 1)  # 将前800个元素作为训练集
model_memory_valid = np.array(Self_CPU_Mem_values[slice_num:]).reshape(runs - slice_num, 1)  # 将后200个元素作为测试集


for file_num in range(1, 19):
    file_path = "data/cpu_memory/batch_1/latency/module_" + str(file_num) + ".txt"
    # Read data from the TXT file and ignore the first line
    with open(file_path, "r") as file:
        lines = file.readlines()[1:(runs + 1)]
    for count, line in enumerate(lines):
        data = line.strip().split()
        if count < slice_num:
            if file_num < 17:
                configs_train[count][file_num * 2 - 2] = float(data[0])
                configs_train[count][file_num * 2 - 1] = float(data[1])
            elif file_num == 17:
                configs_train[count][32] = float(data[0])
                configs_train[count][33] = float(data[1])
                configs_train[count][34] = float(data[2])
            else:
                configs_train[count][35] = float(data[0])
                configs_train[count][36] = float(data[1])
                configs_train[count][37] = float(data[2])


        else:
            if file_num < 17:
                configs_valid[count - slice_num][file_num * 2 - 2] = float(data[0])
                configs_valid[count - slice_num][file_num * 2 - 1] = float(data[1])
            elif file_num == 17:
                configs_valid[count - slice_num][32] = float(data[0])
                configs_valid[count - slice_num][33] = float(data[1])
                configs_valid[count - slice_num][34] = float(data[2])
            else:
                configs_valid[count - slice_num][35] = float(data[0])
                configs_valid[count - slice_num][36] = float(data[1])
                configs_valid[count - slice_num][37] = float(data[2])

configs_train_T = configs_train.T
A1 = np.dot(configs_train_T, configs_train)
A2 = np.linalg.inv(A1)
A3 = np.dot(A2, configs_train_T)
X = np.dot(A3, model_memory_train)

coefficients_list = [X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0], X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0], X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0], X[34, 0], X[35, 0], X[36, 0], X[37, 0]]
print('coefficients_list:\n', coefficients_list)

print(
    '38 configs的多元一次方程平面拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f * config_3 + %.3f * config_4 + %.3f * config_5 + %.3f * config_6 + %.3f * config_7 + %.3f * config_8 + %.3f * config_9 + %.3f * config_10 + %.3f * config_11 + %.3f * config_12 + %.3f * config_13 + %.3f * config_14 + %.3f * config_15 + %.3f * config_16 + %.3f * config_17 + %.3f * config_18 + %.3f * config_19 + %.3f * config_20 + %.3f * config_21 + %.3f * config_22 + %.3f * config_23 + %.3f * config_24 + %.3f * config_25 + %.3f * config_26 + %.3f * config_27 + %.3f * config_28 + %.3f * config_29 + %.3f * config_30 + %.3f * config_31 + %.3f * config_32 + %.3f * config_33 + %.3f * config_34 + %.3f * config_35 + %.3f * config_36 + %.3f * config_37 + %.3f' %
    (X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0],
     X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0],
     X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0],
     X[34, 0], X[35, 0], X[36, 0], X[37, 0]))
model_memory_predict = np.dot(configs_valid, X)
model_memory_acc = 1 - abs(model_memory_predict - model_memory_valid) / model_memory_valid
model_memory_avg_acc = np.mean(model_memory_acc)
model_memory_std_acc = np.std(model_memory_acc)
print("--------------------------------------------------")
print("38 configs的平均准确率为:", model_memory_avg_acc)
print('标准差为:', model_memory_std_acc)
print("--------------------------------------------------")

# Create the plot
fig, ax = plt.subplots()
ax.tick_params(axis='both', labelsize=12)
# Generating x-axis values using the range function
x_values = list(range(1, len(model_memory_acc) + 1))
plt.plot(x_values, model_memory_acc, marker='o', color='green')
plt.xlabel('run period', fontsize=12)
plt.ylabel('model memory acc', fontsize=12)
plt.title('Model Memory Accuracy and Statistics (using all 38 configs)', fontsize=12)

# Add annotations for module_avg_acc and module_std_acc in the lower-left corner
x_coord = min(x_values)  # Using the leftmost x-coordinate
y_coord = min(model_memory_acc)  # Using the lowest y-coordinate
plt.text(x_coord, y_coord, f'acc avg: {model_memory_avg_acc:.3f}', fontsize=12)
plt.text(x_coord + 60, y_coord, f'acc std: {model_memory_std_acc:.3f}', fontsize=12)

plt.grid(True)
plt.tight_layout()  # To improve spacing
plt.show()


############################## box_plot #####################################
print("model_memory_acc is:\n", model_memory_acc)
# Save model_latency_acc to a TXT file
np.savetxt('../box_plot/box_plot_data/jetson_cpu_model_memory_acc.txt', model_memory_acc)

# Plotting the boxplot of model_latency_acc
# fig, ax = plt.subplots()
# ax.tick_params(axis='both', labelsize=12)
plt.boxplot(model_memory_acc, sym='+')
plt.xlabel('jetson cpu', fontsize=12)
plt.ylabel('model memory acc', fontsize=12)
plt.title('Boxplot of Model Memory Accuracy (using all 38 configs)', fontsize=12)

plt.grid(True)
plt.tight_layout()  # To improve spacing
plt.show()
