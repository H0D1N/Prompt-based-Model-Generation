import numpy as np
import re
import matplotlib.pyplot as plt


runs = 500
configs_data = np.ones((runs, 39))

# 从configs.txt文件中提取数值
with open("data/batch_1/configs.txt", 'r') as file:
    for i, line in enumerate(file):

        # 将每一行的前38个数值赋值给configs_train和configs_valid的前38列
        values = list(map(float, line.strip().split()[:38]))

        configs_data[i, :38] = np.array(values)
# Split data into input (configs) and output (latency) variables
configs = configs_data




# 用于存储500个值的列表
peak_memory_footprint = []

file_path = f"data/batch_1/memory_data.txt"
# Read data from the file
memory_data = []
with open(file_path, "r") as f:
    for line in f:
        memory_value = float(line.strip())
        peak_memory_footprint.append(memory_value)



print("Overall peak memory footprint (MB):\n", peak_memory_footprint)
print()

mem = peak_memory_footprint

# # "开启测量mem"情况下的推理延时
# avg_Inference = []
# with open("data/batch_1.txt", 'r') as file:
#     for i, line in enumerate(file):
#         values = line.strip().split()  # Split the line into individual values
#         if len(values) >= 39:  # Ensure the line has at least 39 values
#             Inference_value = float(values[38])  # Read the 39th value (0-based index)
#             avg_Inference.append(Inference_value)
#         else:
#             print(f"Skipping line with fewer than 39 values: {line}")
#
# print(avg_Inference)
# print()


# Define the validation data size
valid_size = 100

# Initialize lists to store accuracy values and training data sizes
accuracy_values = []
train_data_sizes = list(range(50, runs-valid_size+1, 50))  # Train data sizes starting from 50 and increasing by 50

# Loop through different training data sizes
for train_size in train_data_sizes:
    # Select a subset of training data
    configs_train = configs[:train_size]
    mem_train = mem[:train_size]

    # Calculate model parameters using linear regression
    A1 = np.dot(configs_train.T, configs_train)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, configs_train.T)
    X = np.dot(A3, mem_train)

    # Predict mem on validation data
    model_mem_valid = mem[valid_size:]
    model_mem_predict = np.dot(configs[valid_size:], X)

    # Calculate accuracy
    model_mem_acc = 1 - np.abs(model_mem_predict - model_mem_valid) / model_mem_valid
    model_mem_avg_acc = np.mean(model_mem_acc)

    # Store the accuracy value
    accuracy_values.append(model_mem_avg_acc)

# Create the plot
plt.figure()
plt.figure(figsize=(10, 7))  # Increase figure size
plt.plot(train_data_sizes, accuracy_values, marker='o', markersize=10, color='orange', linewidth=4)
plt.xlabel('Number of Random Configurations Generated', fontsize=26)
plt.ylabel('Memory Prediction Acc', fontsize=26)
# Set the font size of x and y axis tick labels
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
# plt.title('Effect of Training Data Size on Model Memory Accuracy', fontsize=12)
plt.grid(True)
plt.tight_layout()  # To improve spacing
plt.show()

# Save accuracy_values to a text file
with open("../../acc vs configs/data/jetson_gpu_memory_accuracy_values.txt", "w") as f:
    for value in accuracy_values:
        f.write("%.4f\n" % value)


print("%f GB" % (np.mean(mem)/1024))  # GB