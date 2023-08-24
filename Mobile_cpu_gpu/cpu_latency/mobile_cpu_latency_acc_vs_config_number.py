import numpy as np
import re
import matplotlib.pyplot as plt

# Load data from the file
runs = 2000
configs_data = np.ones((runs, 39))
model_latency_data = np.zeros((runs, 1))

# 用于存储1000个avg值的列表
avg_Inference = []

# 处理从1.txt到1000.txt的文件
for file_number in range(1, runs+1):
    file_path = f"data_newest/results/{file_number}.txt"

    with open(file_path, 'r') as file:
        log = file.read()
    # 使用正则表达式匹配目标值
    match = re.search(r'Inference \(avg\): ([+-]?\d+(\.\d*)?(e[+-]?\d+)?)', log)

    # 提取目标值
    if match:
        avg_Inference.append(float(match.group(1)))

    else:
        print("未找到目标值")

# 从configs.txt文件中提取数值
with open("../configs.txt", 'r') as file:
    for i, line in enumerate(file):
        # 将每一行的前38个数值赋值给configs_train和configs_valid的前38列
        values = list(map(float, line.strip().split()[:38]))

        configs_data[i, :38] = np.array(values)

# Split data into input (configs) and output (latency) variables
configs = configs_data
latency = avg_Inference
# print(configs)
# print(latency)
# Define the validation data size
valid_size = 400

# Initialize lists to store accuracy values and training data sizes
accuracy_values = []
train_data_sizes = list(range(50, runs - valid_size + 1, 50))  # Train data sizes starting from 50 and increasing by 50

# Loop through different training data sizes
for train_size in train_data_sizes:
    # Select a subset of training data
    configs_train = configs[:train_size]
    latency_train = latency[:train_size]

    # Calculate model parameters using linear regression
    A1 = np.dot(configs_train.T, configs_train)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, configs_train.T)
    X = np.dot(A3, latency_train)

    # Predict latency on validation data
    model_latency_valid = latency[valid_size:]
    model_latency_predict = np.dot(configs[valid_size:], X)

    # Calculate accuracy
    model_latency_acc = 1 - np.abs(model_latency_predict - model_latency_valid) / model_latency_valid
    model_latency_avg_acc = np.mean(model_latency_acc)

    # Store the accuracy value
    accuracy_values.append(model_latency_avg_acc)

# Create the plot
plt.figure()
plt.figure(figsize=(10, 7))  # Increase figure size
plt.plot(train_data_sizes, accuracy_values, marker='o', markersize=10, color='orange', linewidth=4)
plt.xlabel('Number of Random Configurations Generated', fontsize=26)
plt.ylabel('Latency Prediction Acc', fontsize=26)
# Set the font size of x and y axis tick labels
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
# plt.title('Effect of Training Data Size on Model Latency Accuracy', fontsize=12)
plt.grid(True)
plt.tight_layout()  # To improve spacing
plt.show()

# Save accuracy_values to a text file
with open("../../acc vs configs/data/mobile_cpu_latency_accuracy_values.txt", "w") as f:
    for value in accuracy_values:
        f.write("%.4f\n" % value)


print("%f ms" % (np.mean(latency)/1000))  # ms
