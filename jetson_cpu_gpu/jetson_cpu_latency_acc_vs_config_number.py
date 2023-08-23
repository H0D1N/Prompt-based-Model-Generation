import numpy as np
import matplotlib.pyplot as plt

# Load data from the file
runs = 500
configs_data = np.ones((runs, 39))
model_latency_data = np.zeros((runs, 1))
for file_num in range(1, 19):
    file_path = "data/cpu_latency/module_" + str(file_num) + ".txt"
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
            model_latency_data[count] = round(np.mean([float(val) for val in data[-10:]]) / 1000, 3)

# Split data into input (configs) and output (latency) variables
configs = configs_data
latency = model_latency_data
# print(configs)
# print(latency)
# Define the validation data size
valid_size = 100

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
with open("../acc vs configs/data/jetson_cpu_latency_accuracy_values.txt", "w") as f:
    for value in accuracy_values:
        f.write("%.4f\n" % value)


print("%f ms" % (np.mean(latency)))  # ms
