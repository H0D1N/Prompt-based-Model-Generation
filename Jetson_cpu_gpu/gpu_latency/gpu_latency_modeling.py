import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

runs = 500
ratio = 0.8
slice_num = int(runs * ratio)
batch_size = [1]
################################### predictor using all 38 configs #################################
configs_train = np.ones((slice_num, 39))
model_latency_train = np.zeros((slice_num, 1))
configs_valid = np.ones((runs-slice_num, 39))
model_latency_valid = np.zeros((runs-slice_num, 1))

for batch in batch_size:
    file_path = "data/gpu_latency/batch_" + str(batch) + ".txt"
    # Read data from the TXT file and ignore the first line
    with open(file_path, "r") as file:
        lines = file.readlines()[:(runs + 1)]
    for count, line in enumerate(lines):
        data = line.strip().split()
        if count < slice_num:
                configs_train[count][0:38] = [float(x) for x in data[0:38]]
                model_latency_train[count] = float(data[-1])
        else:
                configs_valid[count-slice_num][0:38] = [float(x) for x in data[0:38]]
                model_latency_valid[count-slice_num] = float(data[-1])

    configs_train_T = configs_train.T
    A1 = np.dot(configs_train_T, configs_train)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, configs_train_T)
    X = np.dot(A3, model_latency_train)

    coefficients_list = [X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0], X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0], X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0], X[34, 0], X[35, 0], X[36, 0], X[37, 0]]
    print('coefficients_list:\n', coefficients_list)

    print('38 configs的多元一次方程平面拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f * config_3 + %.3f * config_4 + %.3f * config_5 + %.3f * config_6 + %.3f * config_7 + %.3f * config_8 + %.3f * config_9 + %.3f * config_10 + %.3f * config_11 + %.3f * config_12 + %.3f * config_13 + %.3f * config_14 + %.3f * config_15 + %.3f * config_16 + %.3f * config_17 + %.3f * config_18 + %.3f * config_19 + %.3f * config_20 + %.3f * config_21 + %.3f * config_22 + %.3f * config_23 + %.3f * config_24 + %.3f * config_25 + %.3f * config_26 + %.3f * config_27 + %.3f * config_28 + %.3f * config_29 + %.3f * config_30 + %.3f * config_31 + %.3f * config_32 + %.3f * config_33 + %.3f * config_34 + %.3f * config_35 + %.3f * config_36 + %.3f * config_37 + %.3f' % 
        (X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0], X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0], X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0], X[34, 0], X[35, 0], X[36, 0], X[37, 0]))
    model_latency_predict = np.dot(configs_valid, X)
    model_latency_acc = 1 - abs(model_latency_predict - model_latency_valid) / model_latency_valid
    model_latency_avg_acc = np.mean(model_latency_acc)
    model_latency_std_acc = np.std(model_latency_acc)
    print("--------------------------------------------------")
    print("38 configs拟合多元一次方程并预测的平均准确率为:", model_latency_avg_acc)
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