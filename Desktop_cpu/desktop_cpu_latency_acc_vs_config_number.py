import numpy as np
import matplotlib.pyplot as plt

# Load data from the file
data = np.loadtxt("data/latency_time.txt")

# Split data into input (configs) and output (latency) variables
configs = data[:, :38]
latency = data[:, 38]

# Define the validation data size
valid_size = 200

# Initialize lists to store accuracy values and training data sizes
accuracy_values = []
train_data_sizes = list(range(50, 801, 50))  # Train data sizes starting from 50 and increasing by 50

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
with open("../acc vs configs/data/server_cpu_latency_accuracy_values.txt", "w") as f:
    for value in accuracy_values:
        f.write("%.4f\n" % value)



print("%f ms" % (np.mean(latency)))  # ms
