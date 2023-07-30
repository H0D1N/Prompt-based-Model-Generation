import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
runs = 3000
ratio = 0.7  # 80% of the data is used for training, and 20% is used for testing
slice_num = int(runs * ratio)  # 800

# fig = plt.figure()
# ##################################### modeling 1 to 16  #######################################

# for file_num in range(1, 17):
#     file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
#     # Read data from the TXT file and ignore the first line
#     with open(file_path, "r") as file:
#         lines = file.readlines()[1:(slice_num + 1)]

#     A = np.ones((len(lines), 3))
#     b = np.ones((len(lines), 1))
#     count = 0
#     # Extract data from each line and calculate the average for columns 3, 4, and 5
#     for line in lines:
#         data = line.strip().split()
#         A[count][0] = float(data[0])
#         A[count][1] = float(data[1])
#         b[count] = round(np.mean([float(val) for val in data[2:5]]) / 1000, 3)
#         count += 1

#     A_T = A.T
#     A1 = np.dot(A_T, A)
#     A2 = np.linalg.inv(A1)
#     A3 = np.dot(A2, A_T)
#     X = np.dot(A3, b)
#     print('第%d个bottleneck的二元一次方程平面拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f' % (
#         file_num, X[0, 0], X[1, 0], X[2, 0]))

#     # 计算方差
#     R = 0
#     for i in range(0, slice_num):
#         R = R + (X[0, 0] * A[i, 0] + X[1, 0] * A[i, 1] + X[2, 0] - b[i]) ** 2
#     std_R = np.sqrt(R / slice_num)
#     print('标准差为:', std_R)
#     print("--------------------------------------------------")

#     # 展示图像
#     ax1 = fig.add_subplot(4, 4, file_num, projection='3d')
#     ax1.tick_params(axis='both', labelsize = 8)
#     ax1.set_xlabel("config_1", fontsize=8)
#     ax1.set_ylabel("config_2", fontsize=8)
#     ax1.set_zlabel("time(ms)", fontsize=8)
#     ax1.scatter(A[:, 0], A[:, 1], b, c='r', marker='o')
#     ax1.set_title("bottleneck_" + str(file_num) + "_std_" + str(std_R), fontsize=8)

#     x_p = np.linspace(0, 1, 300)
#     y_p = np.linspace(0, 1, 300)
#     x_p, y_p = np.meshgrid(x_p, y_p)
#     z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
#     ax1.plot_wireframe(x_p, y_p, z_p, rstride=30, cstride=30)

# # 调整子图之间的间距
# plt.tight_layout()
# plt.show()

# ##################################### modeling 17 and 18 #######################################


# for file_num in range(17, 19):  # Process files 17 and 18
#     file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
#     # Read data from the TXT file and ignore the first line
#     with open(file_path, "r") as file:
#         lines = file.readlines()[1:(slice_num + 1)]

#     A = np.ones((len(lines), 4))
#     b = np.ones((len(lines), 1))
#     count = 0
#     # Extract data from each line and calculate the average for columns 3, 4, and 5
#     for line in lines:
#         data = line.strip().split()
#         A[count][0] = float(data[0])
#         A[count][1] = float(data[1])
#         A[count][2] = float(data[2])
#         b[count] = round(np.mean([float(val) for val in data[3:6]]) / 1000, 3)
#         count += 1

#     # Assuming you have already loaded and prepared the data as shown in your code.

#     A_T = A.T
#     A1 = np.dot(A_T, A)
#     A2 = np.linalg.inv(A1)
#     A3 = np.dot(A2, A_T)
#     X = np.dot(A3, b)
#     print(
#         '第%d个module的三元一次方程拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f * config_3 + %.3f' % (
#             file_num, X[0, 0], X[1, 0], X[2, 0], X[3, 0]))

#     # 计算方差
#     R = 0
#     for i in range(0, slice_num):
#         R = R + (X[0, 0] * A[i, 0] + X[1, 0] * A[i, 1] + X[2, 0] * A[i, 2] + X[3, 0] -
#                  b[i]) ** 2
#     std_R = np.sqrt(R / slice_num)
#     print('标准差为:', std_R)
#     print("--------------------------------------------------")


# ################################# Predictor ######################################

# def predictor(file_num):
#     # Coefficients obtained from the fitting process
#     ''' runs组训练
#     coefficients = {
#         1: [5.167, 5.702, 16.679],
#         2: [6.908, 6.339, 8.988],
#         3: [6.856, 6.072, 9.306],
#         4: [9.699, 5.307, 13.700],
#         5: [5.958, 5.608, 2.655],
#         6: [5.921, 5.424, 2.836],
#         7: [6.812, 5.813, 11.008],
#         8: [8.775, 5.270, 8.255],
#         9: [5.433, 4.978, 1.066],
#         10: [5.333, 5.197, 1.011],
#         11: [5.508, 5.214, 0.916],
#         12: [5.297, 4.926, 1.218],
#         13: [5.382, 5.009, 1.111],
#         14: [8.071, 5.819, 6.560],
#         15: [5.981, 5.765, 0.035],
#         16: [5.851, 5.616, 0.187],
#         17: [6.455, 4.858, 3.188, 2.718],
#         18: [6.150, 4.961, 5.174, 2.371]
#     }
#     '''

#     # 800组训练
#     coefficients = {
#         1: [5.130, 5.846, 16.663],
#         2: [6.944, 6.307, 8.975],
#         3: [6.817, 6.069, 9.350],
#         4: [9.762, 5.267, 13.700],
#         5: [5.892, 5.548, 2.659],
#         6: [5.902, 5.357, 2.843],
#         7: [6.932, 5.947, 10.788],
#         8: [8.760, 5.308, 8.196],
#         9: [5.420, 4.980, 1.026],
#         10: [5.347, 5.226, 0.956],
#         11: [5.497, 5.215, 0.884],
#         12: [5.267, 4.861, 1.238],
#         13: [5.407, 4.973, 1.085],
#         14: [8.051, 5.829, 6.544],
#         15: [5.986, 5.793, -0.013],
#         16: [5.893, 5.628, 0.126],
#         17: [6.626, 4.876, 3.187, 2.442],
#         18: [6.094, 4.906, 5.167, 2.273]
#     }

#     if 1 <= file_num <= 16:

#         file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
#         # Read data from the TXT file and ignore the first line
#         with open(file_path, "r") as file:
#             lines = file.readlines()[(slice_num + 1):]

#         module_config = np.ones((len(lines), 3))
#         module_predict = np.zeros((len(lines), 1))
#         module_real = np.zeros((len(lines), 1))


#         count = 0
#         # Extract data from each line and calculate the average for columns 3, 4, and 5
#         for line in lines:
#             data = line.strip().split()
#             module_config[count][0] = float(data[0])
#             module_config[count][1] = float(data[1])
#             module_real[count] = round(np.mean([float(val) for val in data[2:5]]) / 1000, 3)
#             module_predict[count] = module_config[count][0] * coefficients[file_num][0] + module_config[count][1] * coefficients[file_num][1] + \
#                        coefficients[file_num][2]
#             count += 1

#         # print("Predicted module_predict value for 2-element config:\n", module_predict)
#         return module_predict, module_real

#     if file_num == 17 or file_num == 18:

#         file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
#         # Read data from the TXT file and ignore the first line
#         with open(file_path, "r") as file:
#             lines = file.readlines()[(slice_num + 1):]

#         module_config = np.ones((len(lines), 4))
#         module_predict = np.zeros((len(lines), 1))
#         module_real = np.zeros((len(lines), 1))

#         count = 0
#         # Extract data from each line and calculate the average for columns 3, 4, and 5
#         for line in lines:
#             data = line.strip().split()
#             module_config[count][0] = float(data[0])
#             module_config[count][1] = float(data[1])
#             module_config[count][2] = float(data[2])
#             module_real[count] = round(np.mean([float(val) for val in data[3:6]]) / 1000, 3)
#             module_predict[count] = module_config[count][0] * coefficients[file_num][0] + module_config[count][1] * coefficients[file_num][1] + \
#                        module_config[count][2] * coefficients[file_num][2] + coefficients[file_num][3]
#             count += 1

#         # print("Predicted module_predict value for 3-element config", module_predict)
#         return module_predict, module_real


# #################################### module_evaluator ####################################

# def module_evaluator(file_num):
#     module_predict, module_real = predictor(file_num)

#     module_acc = 1 - abs(module_predict - module_real) / module_real
#     module_avg_acc = np.mean(module_acc)
#     module_std_acc = np.std(module_acc)
#     return module_acc, module_avg_acc, module_std_acc


# # Create a 3x6 grid of subplots
# fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 12))
# for ax in axes.flat:
#     ax.tick_params(axis='both', labelsize=8)
# # Example usage:
# for file_num in range(1, 19):
#     module_acc, module_avg_acc, module_std_acc = module_evaluator(file_num)

#     # Determine the subplot index based on file_num
#     row = (file_num - 1) // 6
#     col = (file_num - 1) % 6

#     # Generating x-axis values using the range function
#     x_values = list(range(1, len(module_acc) + 1))

#     # Plot the data in the corresponding subplot
#     axes[row, col].plot(x_values, module_acc, marker='o', color='blue', linewidth=0.3, markersize=1)
#     axes[row, col].set_xlabel('run_period', fontsize=12)
#     axes[row, col].set_ylabel('acc', fontsize=12)
#     axes[row, col].set_title(f'module {file_num} acc', fontsize=12)

#     # Add annotations for module_avg_acc and module_std_acc in the lower-left corner
#     x_coord = min(x_values)  # Using the leftmost x-coordinate
#     y_coord = min(module_acc)  # Using the lowest y-coordinate
#     axes[row, col].text(x_coord, y_coord, f'Avg. Acc.: {module_avg_acc:.3f}', fontsize=8)
#     axes[row, col].text(x_coord + 100, y_coord, f'Std. Acc.: {module_std_acc:.3f}', fontsize=8)
# # Adjust layout and spacing
# plt.tight_layout()
# # Show the figure
# plt.show()


# #################### avg_overhead_total_model_latency #######################

# def avg_overhead_model_latency():
#     ############# Extract total model latency ###########
#     file_path = "data/modeling/batch_1/module_" + str(1) + ".txt"
#     # Read data from the TXT file and ignore the first line
#     with open(file_path, "r") as file:
#         lines = file.readlines()[1:]

#     model_latency = np.zeros((len(lines), 1))

#     # Extract data from each line and calculate the average for columns 6, 7, and 8
#     for count, line in enumerate(lines):
#         data = line.strip().split()
#         model_latency[count] = round(np.mean([float(val) for val in data[5:8]]) / 1000, 3)

#     ####### Extract sum (module latency) ###############
#     sum_module = np.zeros((len(lines), 1))
#     for file_num in range(1, 17):
#         file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
#         # Read data from the TXT file and ignore the first line
#         with open(file_path, "r") as file:
#             lines = file.readlines()[1:]

#         tmp = np.zeros((len(lines), 1))
#         # Extract data from each line and calculate the average for columns 3, 4, and 5
#         for count, line in enumerate(lines):
#             data = line.strip().split()
#             tmp[count] = round(np.mean([float(val) for val in data[2:5]]) / 1000, 3)
        
#         sum_module += tmp

#     for file_num in range(17, 19):
#         file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
#         # Read data from the TXT file and ignore the first line
#         with open(file_path, "r") as file:
#             lines = file.readlines()[1:]

#         tmp = np.zeros((len(lines), 1))
#         # Extract data from each line and calculate the average for columns 4, 5, and 6
#         for count, line in enumerate(lines):
#             data = line.strip().split()
#             tmp[count] = round(np.mean([float(val) for val in data[3:6]]) / 1000, 3)
    
#         sum_module += tmp

#     # overhead = total latency - sum (module latency)
#     total_overhead = model_latency - sum_module
#     avg_overhead = np.mean(total_overhead[:(slice_num + 1)])
#     print("Average overhead over ", ratio, " data for training is ",avg_overhead, "ms")

#     return avg_overhead, model_latency


# ################################### total_model_evaluator #################################

# def total_model_evaluator():
#     avg_model_overhead, model_latency = avg_overhead_model_latency()

#     total_module_predict = np.zeros((runs - slice_num, 1))

#     for file_num in range(1, 19):
#         module_predict, _ = predictor(file_num)

#         total_module_predict += module_predict

#     total_model_acc = 1 - abs(total_module_predict + avg_model_overhead - model_latency[slice_num:]) / model_latency[slice_num:]
#     total_model_avg_acc = np.mean(total_model_acc)
#     total_model_std_acc = np.std(total_model_acc)
#     return total_model_acc, total_model_avg_acc, total_model_std_acc


# # Call the function to get the data
# total_model_acc, total_model_avg_acc, total_model_std_acc = total_model_evaluator()

# # Create the plot
# fig, ax = plt.subplots()
# ax.tick_params(axis='both', labelsize=12)
# # Generating x-axis values using the range function
# x_values = list(range(1, len(total_model_acc) + 1))
# plt.plot(x_values, total_model_acc, marker='o', color='blue')

# # Add annotations for module_avg_acc and module_std_acc in the lower-left corner
# x_coord = min(x_values)  # Using the leftmost x-coordinate
# y_coord = min(total_model_acc)  # Using the lowest y-coordinate
# plt.text(x_coord, y_coord, f'acc avg: {total_model_avg_acc:.3f}', fontsize=12)
# plt.text(x_coord + 50, y_coord, f'acc std: {total_model_std_acc:.3f}', fontsize=12)

# plt.xlabel('run period', fontsize=12)
# plt.ylabel('model prediction acc', fontsize=12)
# plt.title('Model Prediction Accuracy and Statistics', fontsize=12)
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()

################################### predictor using all 38 configs #################################
configs_train = np.ones((slice_num, 39))
model_latency_train = np.zeros((slice_num,1))
configs_valid = np.ones((runs-slice_num, 39))
model_latency_valid = np.zeros((runs-slice_num,1))

for file_num in range(1, 19):
    file_path = "data/modeling/batch_1/module_" + str(file_num) + ".txt"
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
                model_latency_train[count] = round(np.mean([float(val) for val in data[-3:]]) / 1000, 3)
            
        else:
            if file_num < 17:
                configs_valid[count-slice_num][file_num * 2 - 2] = float(data[0])
                configs_valid[count-slice_num][file_num * 2 - 1] = float(data[1])
            elif file_num == 17:
                configs_valid[count-slice_num][32] = float(data[0])
                configs_valid[count-slice_num][33] = float(data[1])
                configs_valid[count-slice_num][34] = float(data[2])
            else:
                configs_valid[count-slice_num][35] = float(data[0])
                configs_valid[count-slice_num][36] = float(data[1])
                configs_valid[count-slice_num][37] = float(data[2])
                model_latency_valid[count-slice_num] = round(np.mean([float(val) for val in data[-3:]]) / 1000, 3)

configs_train_T = configs_train.T
A1 = np.dot(configs_train_T, configs_train)
A2 = np.linalg.inv(A1)
A3 = np.dot(A2, configs_train_T)
X = np.dot(A3, model_latency_train)

print('38 configs的多元一次方程平面拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f * config_3 + %.3f * config_4 + %.3f * config_5 + %.3f * config_6 + %.3f * config_7 + %.3f * config_8 + %.3f * config_9 + %.3f * config_10 + %.3f * config_11 + %.3f * config_12 + %.3f * config_13 + %.3f * config_14 + %.3f * config_15 + %.3f * config_16 + %.3f * config_17 + %.3f * config_18 + %.3f * config_19 + %.3f * config_20 + %.3f * config_21 + %.3f * config_22 + %.3f * config_23 + %.3f * config_24 + %.3f * config_25 + %.3f * config_26 + %.3f * config_27 + %.3f * config_28 + %.3f * config_29 + %.3f * config_30 + %.3f * config_31 + %.3f * config_32 + %.3f * config_33 + %.3f * config_34 + %.3f * config_35 + %.3f * config_36 + %.3f * config_37 + %.3f' % 
    (X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0], X[6, 0], X[7, 0], X[8, 0], X[9, 0], X[10, 0], X[11, 0], X[12, 0], X[13, 0], X[14, 0], X[15, 0], X[16, 0], X[17, 0], X[18, 0], X[19, 0], X[20, 0], X[21, 0], X[22, 0], X[23, 0], X[24, 0], X[25, 0], X[26, 0], X[27, 0], X[28, 0], X[29, 0], X[30, 0], X[31, 0], X[32, 0], X[33, 0], X[34, 0], X[35, 0], X[36, 0], X[37, 0]))
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