import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D

runs = 1000
ratio = 0.8  # 80% of the data is used for training, and 20% is used for testing
slice_num = int(runs * ratio)  # 800


# 用于存储1000个avg值的列表
avg_Inference = []

# 处理从1.txt到1000.txt的文件
for file_number in range(1, 1001):
    file_path = f"data/gpu_latency/results/{file_number}.txt"

    with open(file_path, 'r') as file:
        log = file.read()
    # 使用正则表达式匹配目标值
    match = re.search(r'Inference \(avg\): ([+-]?\d+(\.\d*)?(e[+-]?\d+)?)', log)

    # 提取目标值
    if match:
        avg_Inference.append(float(match.group(1)))

    else:
        print("未找到目标值")

print("avg_Inference (us):\n", avg_Inference)
print()

################################### memory predictor using all 38 configs #################################

model_latency_train = np.array(avg_Inference[:slice_num]).reshape(slice_num, 1)  # 将前800个元素作为训练集
model_latency_valid = np.array(avg_Inference[slice_num:]).reshape(runs - slice_num, 1)  # 将后200个元素作为测试集


# 初始化configs_train和configs_valid
configs_train = np.ones((slice_num, 39))
configs_valid = np.ones((runs - slice_num, 39))

# 从configs.txt文件中提取数值并赋值给configs_train和configs_valid
with open("../onnx_models_op12/configs.txt", 'r') as file:
    for i, line in enumerate(file):
        # 将每一行的前38个数值赋值给configs_train和configs_valid的前38列
        values = list(map(float, line.strip().split()[:38]))
        if i < slice_num:
            configs_train[i, :38] = np.array(values)
        else:
            configs_valid[i - slice_num, :38] = np.array(values)

    # 打印结果（可选）
    print("configs_train:\n", configs_train)
    print("configs_valid:\n", configs_valid)

configs_train_T = configs_train.T
A1 = np.dot(configs_train_T, configs_train)
A2 = np.linalg.inv(A1)
A3 = np.dot(A2, configs_train_T)
X = np.dot(A3, model_latency_train)

coefficients_list = [X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0], X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0], X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0], X[34, 0], X[35, 0], X[36, 0], X[37, 0]]
print('coefficients_list:\n', coefficients_list)

print(
    '38 configs的多元一次方程平面拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f * config_3 + %.3f * config_4 + %.3f * config_5 + %.3f * config_6 + %.3f * config_7 + %.3f * config_8 + %.3f * config_9 + %.3f * config_10 + %.3f * config_11 + %.3f * config_12 + %.3f * config_13 + %.3f * config_14 + %.3f * config_15 + %.3f * config_16 + %.3f * config_17 + %.3f * config_18 + %.3f * config_19 + %.3f * config_20 + %.3f * config_21 + %.3f * config_22 + %.3f * config_23 + %.3f * config_24 + %.3f * config_25 + %.3f * config_26 + %.3f * config_27 + %.3f * config_28 + %.3f * config_29 + %.3f * config_30 + %.3f * config_31 + %.3f * config_32 + %.3f * config_33 + %.3f * config_34 + %.3f * config_35 + %.3f * config_36 + %.3f * config_37 + %.3f' %
    (X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0],
     X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0],
     X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0],
     X[34, 0], X[35, 0], X[36, 0], X[37, 0]))
model_latency_predict = np.dot(configs_valid, X)
model_latency_acc = 1 - abs(model_latency_predict - model_latency_valid) / model_latency_valid
model_latency_avg_acc = np.mean(model_latency_acc)
model_latency_std_acc = np.std(model_latency_acc)
print("--------------------------------------------------")
print("38 configs的平均准确率为:", model_latency_avg_acc)
print('标准差为:', model_latency_std_acc)
print("--------------------------------------------------")

# Create the plot
fig, ax = plt.subplots()
ax.tick_params(axis='both', labelsize=12)
# Generating x-axis values using the range function
x_values = list(range(1, len(model_latency_acc) + 1))
plt.plot(x_values, model_latency_acc, marker='o', color='blue')
plt.xlabel('run period', fontsize=12)
plt.ylabel('model latency acc', fontsize=12)
plt.title('Model Latency Accuracy and Statistics (using all 38 configs)', fontsize=12)

# Add annotations for module_avg_acc and module_std_acc in the lower-left corner
x_coord = min(x_values)  # Using the leftmost x-coordinate
y_coord = min(model_latency_acc)  # Using the lowest y-coordinate
plt.text(x_coord, y_coord, f'acc avg: {model_latency_avg_acc:.3f}', fontsize=12)
plt.text(x_coord + 100, y_coord, f'acc std: {model_latency_std_acc:.3f}', fontsize=12)

plt.grid(True)
plt.tight_layout()  # To improve spacing
plt.show()


############################## box_plot #####################################
print("model_latency_acc is:\n", model_latency_acc)
# Save model_latency_acc to a TXT file
np.savetxt('../../box_plot/box_plot_data/mobile_gpu_tflite_model_latency_acc.txt', model_latency_acc)

# Plotting the boxplot of model_latency_acc
# fig, ax = plt.subplots()
# ax.tick_params(axis='both', labelsize=12)
plt.boxplot(model_latency_acc, sym='+')
plt.xlabel('mobile_gpu_tflite', fontsize=12)
plt.ylabel('model latency acc', fontsize=12)
plt.title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)

plt.grid(True)
plt.tight_layout()  # To improve spacing
plt.show()