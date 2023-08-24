import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# '''Latency box_plot'''
# ################################# server_cpu_latency_using_time()##################################
# server_cpu_model_latency_acc = np.loadtxt('box_plot_data/server_cpu_model_latency_acc.txt')
#
# plt.boxplot(server_cpu_model_latency_acc, sym='+')
# plt.xlabel('server cpu', fontsize=12)
# plt.ylabel('model latency acc', fontsize=12)
# plt.title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()
#
#
# ################################ jetson_cpu_latency_using_pytorch_profiler ########################
# jetson_cpu_model_latency_acc = np.loadtxt('box_plot_data/jetson_cpu_model_latency_acc.txt')
#
# plt.boxplot(jetson_cpu_model_latency_acc, sym='+')
# plt.xlabel('jetson_cpu', fontsize=12)
# plt.ylabel('model latency acc', fontsize=12)
# plt.title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()
#
#
# ################################ jetson_gpu_latency_using_pytorch_profiler ########################
#
# jetson_gpu_model_latency_acc = np.loadtxt('box_plot_data/jetson_gpu_model_latency_acc.txt')
#
#
# plt.boxplot(jetson_gpu_model_latency_acc, sym='+')
# plt.xlabel('jetson_gpu', fontsize=12)
# plt.ylabel('model latency acc', fontsize=12)
# plt.title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()
#
#
# ################################## mobile_cpu_latency_using_tflite ####################################
#
# mobile_cpu_tflite_model_latency_acc = np.loadtxt('box_plot_data/mobile_cpu_tflite_model_latency_acc.txt')
#
# plt.boxplot(mobile_cpu_tflite_model_latency_acc, sym='+')
# plt.xlabel('mobile_cpu_tflite', fontsize=12)
# plt.ylabel('model latency acc', fontsize=12)
# plt.title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()
#
# ################################## mobile_gpu_latency_using_tflite ####################################
#
# mobile_gpu_tflite_model_latency_acc = np.loadtxt('box_plot_data/mobile_gpu_tflite_model_latency_acc.txt')
#
# plt.boxplot(mobile_gpu_tflite_model_latency_acc, sym='+')
# plt.xlabel('mobile_gpu_tflite', fontsize=12)
# plt.ylabel('model latency acc', fontsize=12)
# plt.title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()


# Load the data from files
server_cpu_model_latency_acc = np.loadtxt('box_plot_data/server_cpu_model_latency_acc.txt')
# jetson_cpu_model_latency_acc = np.loadtxt('box_plot_data/jetson_cpu_model_latency_acc.txt')
jetson_gpu_model_latency_acc = np.loadtxt('box_plot_data/jetson_gpu_model_latency_acc.txt')
mobile_cpu_tflite_model_latency_acc = np.loadtxt('box_plot_data/mobile_cpu_tflite_model_latency_acc.txt')
mobile_gpu_tflite_model_latency_acc = np.loadtxt('box_plot_data/mobile_gpu_tflite_model_latency_acc.txt')

# Combine all data into a list
data = [
    server_cpu_model_latency_acc,
    # jetson_cpu_model_latency_acc,
    jetson_gpu_model_latency_acc,
    mobile_cpu_tflite_model_latency_acc,
    mobile_gpu_tflite_model_latency_acc,
]

# Device labels for the x-axis
devices = [
    'Desktop CPU',
    # 'Jetson CPU',
    'Jetson GPU',
    'Mobile CPU',
    'Mobile GPU',
]

# Create the figure and axes
fig, ax = plt.subplots()
# Plot the box plots with shared y-axis
ax.boxplot(data, sym='+', positions=range(1, len(data) + 1))
# Set x-axis labels
ax.set_xticks(range(1, len(devices) + 1))
ax.set_xticklabels(devices, rotation=30, fontsize=16, fontweight='bold')
# Set y-axis label
ax.set_ylabel('Latency Prediction Acc', fontsize=20, fontweight='bold')
ax.tick_params(axis='y', labelsize=20)

# Set title
# ax.set_title('Boxplot of Model Latency Accuracy (using all 38 configs)', fontsize=12)
# Show the grid
ax.grid(True)
# To improve spacing
plt.tight_layout()
# Show the combined plot
plt.show()

'''memory box_plot'''

# ########################### server_cpu_memory_using_pytorch_profiler #######################
# server_cpu_model_memory_acc = np.loadtxt('box_plot_data/server_cpu_model_memory_acc.txt')
#
# plt.boxplot(server_cpu_model_memory_acc, sym='+')
# plt.xlabel('server cpu', fontsize=12)
# plt.ylabel('model memory acc', fontsize=12)
# plt.title('Boxplot of Model Memory Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()
#
#
# ################################ jetson_cpu_memory_using_pytorch_profiler ########################
# jetson_cpu_model_memory_acc = np.loadtxt('box_plot_data/jetson_cpu_model_memory_acc.txt')
#
# plt.boxplot(jetson_cpu_model_memory_acc, sym='+')
# plt.xlabel('jetson cpu', fontsize=12)
# plt.ylabel('model memory acc', fontsize=12)
# plt.title('Boxplot of Model Memory Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()
#
#
# ################################ mobile_cpu_memory_using_tflite ########################
# mobile_cpu_tflite_model_memory_acc = np.loadtxt('box_plot_data/mobile_cpu_tflite_model_memory_acc.txt')
#
#
# plt.boxplot(mobile_cpu_tflite_model_memory_acc, sym='+')
# plt.xlabel('mobile_cpu_tflite', fontsize=12)
# plt.ylabel('model memory acc', fontsize=12)
# plt.title('Boxplot of Model Memory Accuracy (using all 38 configs)', fontsize=12)
#
# plt.grid(True)
# plt.tight_layout()  # To improve spacing
# plt.show()


# Load the data from files
server_cpu_model_memory_acc = np.loadtxt('box_plot_data/server_cpu_model_memory_acc.txt')
jetson_gpu_model_memory_acc = np.loadtxt('box_plot_data/jetson_gpu_model_memory_acc.txt')
mobile_cpu_tflite_model_memory_acc = np.loadtxt('box_plot_data/mobile_cpu_tflite_model_memory_acc.txt')
mobile_gpu_tflite_model_memory_acc = np.loadtxt('box_plot_data/mobile_gpu_tflite_model_memory_acc.txt')

# Combine all data into a list
data = [
    server_cpu_model_memory_acc,
    jetson_gpu_model_memory_acc,
    mobile_cpu_tflite_model_memory_acc,
    mobile_gpu_tflite_model_memory_acc,
]

# Device labels for the x-axis
devices = [
    'Desktop CPU',
    'Jetson GPU',
    'Mobile CPU',
    'Mobile GPU',
]

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the box plots with shared y-axis
ax.boxplot(data, sym='+', positions=range(1, len(data) + 1))

# Set x-axis labels
ax.set_xticks(range(1, len(devices) + 1))
ax.set_xticklabels(devices, rotation=30, fontsize=16, fontweight='bold')

# Set y-axis label
ax.set_ylabel('Memory Prediction Acc', fontsize=20, fontweight='bold')
ax.tick_params(axis='y', labelsize=20)
# Set title
# ax.set_title('Boxplot of Model Memory Accuracy (using all 38 configs)', fontsize=12)

# Show the grid
ax.grid(True)

# To improve spacing
plt.tight_layout()

# Show the combined plot
plt.show()
