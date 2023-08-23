import matplotlib.pyplot as plt

# File paths
file_paths = [
    "data/server_cpu_latency_accuracy_values.txt",
    "data/server_cpu_memory_accuracy_values.txt",
    "data/jetson_cpu_latency_accuracy_values.txt",
    "data/jetson_cpu_memory_accuracy_values.txt",
    "data/jetson_gpu_latency_accuracy_values.txt",
    "data/mobile_cpu_latency_accuracy_values.txt",
    "data/mobile_cpu_memory_accuracy_values.txt",
    "data/mobile_gpu_latency_accuracy_values.txt",

]

label_names = [
    "server_cpu_latency",
    "server_cpu_memory",
    "jetson_cpu_latency",
    "jetson_cpu_memory",
    "jetson_gpu_latency",
    "mobile_cpu_latency",
    "mobile_cpu_memory",
    "mobile_gpu_latency"
]

# Initialize data for plotting
data_to_plot = []

# Loop through each file
for file_path in file_paths:
    with open(file_path, "r") as f:
        accuracy_values = [float(line.strip()) for line in f]
        data_to_plot.append(accuracy_values)

# Determine x-axis values for each group of data
x_values = [
    list(range(50, 801, 50)),  # For jetson related files
    list(range(50, 801, 50)),  # For other files
    list(range(50, 401, 50)),  # For other files
    list(range(50, 401, 50)),  # For other files
    list(range(50, 401, 50)),  # For other files
    list(range(50, 801, 50)),  # For other files
    list(range(50, 801, 50)),  # For other files
    list(range(50, 801, 50)),  # For other files
]

# Create the plot
plt.figure(figsize=(10, 7))  # Increase figure size

# Plot each group of data
for i in range(len(file_paths)):
    plt.plot(x_values[i], data_to_plot[i], label=label_names[i], marker='o', markersize=6, linewidth=4)

plt.xlabel('Number of Configurations Generated', fontsize=14)
plt.ylabel('Prediction Accuracy', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)

# Set the x-axis limit to start from 100
plt.xlim(100, )
plt.tight_layout()

# Show the plot
plt.show()
