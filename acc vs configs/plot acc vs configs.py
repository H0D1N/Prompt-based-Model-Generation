import matplotlib.pyplot as plt

# File paths
file_paths = [
    "data/server_cpu_latency_accuracy_values.txt",
    "data/server_cpu_memory_accuracy_values.txt",
    "data/jetson_gpu_latency_accuracy_values.txt",
    "data/jetson_gpu_memory_accuracy_values.txt",
    "data/mobile_cpu_latency_accuracy_values.txt",
    "data/mobile_cpu_memory_accuracy_values.txt",
    "data/mobile_gpu_latency_accuracy_values.txt",
    "data/mobile_gpu_memory_accuracy_values.txt",

]

label_names = [
    "Desktop CPU Latency",
    "Desktop CPU Memory",
    "Jetson GPU Latency",
    "Jetson GPU Memory",
    "Mobile CPU Latency",
    "Mobile CPU Memory",
    "Mobile GPU Latency",
    "Mobile GPU Memory"
]

# Define colors for each device
device_colors = ['blue', 'blue', 'red', 'red', 'orange', 'orange', 'brown', 'brown']

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
    list(range(50, 1601, 50)),  # For other files
    list(range(50, 801, 50)),  # For other files
    list(range(50, 1601, 50)),  # For other files
    list(range(50, 1601, 50)),  # For other files
]

# Create the plot
plt.figure(figsize=(7, 6))  # Increase figure size



# Plot each group of data
for i in range(len(file_paths)):
    linestyle = '-' if 'Latency' in label_names[i] else '--' # Use solid line for latency, dashed line for memory
    dashes = (2, 1) if linestyle == '--' else (5, 0)  # Set dashes for dashed line (memory)
    plt.plot(x_values[i], data_to_plot[i], label=label_names[i], marker='o', markersize=6, linewidth=3, color=device_colors[i], linestyle=linestyle, dashes=dashes)
plt.xlabel('Number of Subnets for Training', fontsize=20, fontweight='bold')
plt.ylabel('Prediction Accuracy', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)

# Move the legend to the right side and make the font bold
legend = plt.legend(loc='lower right', fontsize=16)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set the x-axis limit to start from 100
plt.xlim(100, )
plt.tight_layout()

# Show the plot
plt.show()