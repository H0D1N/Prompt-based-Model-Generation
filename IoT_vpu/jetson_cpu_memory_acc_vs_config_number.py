import numpy as np
import matplotlib.pyplot as plt



runs = 500
configs_data = np.ones((runs, 39))
for file_num in range(1, 19):
    file_path = "data/cpu_memory/batch_1/latency/module_" + str(file_num) + ".txt"
    # Read data from the TXT file and ignore the first line
    with open(file_path, "r") as file:
        lines = file.readlines()[1:(runs + 1)]
    for count, line in enumerate(lines):
        data = line.strip().split()
        if file_num < 17:
            configs_data[count][file_num * 2 - 2] = float(data[0])
            configs_data[count][file_num * 2 - 1] = float(data[1])
        elif file_num == 17:
            configs_data[count][32] = float(data[0])
            configs_data[count][33] = float(data[1])
            configs_data[count][34] = float(data[2])
        else:
            configs_data[count][35] = float(data[0])
            configs_data[count][36] = float(data[1])
            configs_data[count][37] = float(data[2])


# Split data into input (configs) and output (latency) variables
configs = configs_data

# 创建一个空列表来存储Self CPU Mem值
Self_CPU_Mem_values = []

# 读取500个txt文件
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



mem = Self_CPU_Mem_values

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
with open("../acc vs configs/data/jetson_cpu_memory_accuracy_values.txt", "w") as f:
    for value in accuracy_values:
        f.write("%.4f\n" % value)


print("%f GB" % (np.mean(mem)))  # GB