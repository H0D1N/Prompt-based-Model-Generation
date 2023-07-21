import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 6})
fig = plt.figure()


##################################### 1 to 16 #######################################

for file_num in range(1, 17):
    file_path = "continuous/module_" + str(file_num) + ".txt"
    # Read data from the TXT file and ignore the first line
    with open(file_path, "r") as file:
        lines = file.readlines()[1:]

    A = np.ones((len(lines), 3))
    b = np.ones((len(lines), 1))
    count = 0
    # Extract data from each line and calculate the average for columns 3, 4, and 5
    for line in lines:
        data = line.strip().split()
        A[count][0] = float(data[0])
        A[count][1] = float(data[1])
        b[count] = round(np.mean([float(val) for val in data[2:5]]) / 1000, 3)
        count += 1

    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    X = np.dot(A3, b)
    print('第%d个bottleneck的二元一次方程平面拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f' % (
        file_num, X[0, 0], X[1, 0], X[2, 0]))

    # 计算标准差
    R = 0
    for i in range(0, len(lines)):
        R = R + (X[0, 0] * A[i, 0] + X[1, 0] * A[i, 1] + X[2, 0] - b[i]) ** 2
    std_R = np.sqrt(R / len(lines))
    print('标准差为:', std_R)
    print("--------------------------------------------------")

    # 展示图像
    ax1 = fig.add_subplot(4, 4, file_num, projection='3d')
    ax1.set_xlabel("config_1")
    ax1.set_ylabel("config_2")
    ax1.set_zlabel("time(ms)")
    ax1.scatter(A[:, 0], A[:, 1], b, c='r', marker='o')
    ax1.set_title("bottleneck_" + str(file_num) + "_std_" + str(std_R))

    x_p = np.linspace(0, 1, 300)
    y_p = np.linspace(0, 1, 300)
    x_p, y_p = np.meshgrid(x_p, y_p)
    z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
    ax1.plot_wireframe(x_p, y_p, z_p, rstride=30, cstride=30)

# 调整子图之间的间距
plt.tight_layout()
plt.show()

##################################### 17 and 18 #######################################


for file_num in range(17, 19):  # Process files 17 and 18
    file_path = "continuous/module_" + str(file_num) + ".txt"
    # Read data from the TXT file and ignore the first line
    with open(file_path, "r") as file:
        lines = file.readlines()[1:]

    A = np.ones((len(lines), 4))
    b = np.ones((len(lines), 1))
    count = 0
    # Extract data from each line and calculate the average for columns 3, 4, and 5
    for line in lines:
        data = line.strip().split()
        A[count][0] = float(data[0])
        A[count][1] = float(data[1])
        A[count][2] = float(data[2])
        b[count] = round(np.mean([float(val) for val in data[3:6]]) / 1000, 3)
        count += 1

    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    X = np.dot(A3, b)
    print(
        '第%d个bottleneck的三元一次方程拟合结果为: z = %.3f * config_1 + %.3f * config_2 + %.3f * config_3 + %.3f' % (
            file_num, X[0, 0], X[1, 0], X[2, 0], X[3, 0]))

    # 计算标准差
    R = 0
    for i in range(0, len(lines)):
        R = R + (X[0, 0] * A[i, 0] + X[1, 0] * A[i, 1] + X[2, 0] * A[i, 2] + X[3, 0] - b[i]) ** 2
    std_R = np.sqrt(R / len(lines))
    print('标准差为:', std_R)
    print("--------------------------------------------------")
