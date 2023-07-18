import torch
import re
import math
import numpy as np
import matplotlib.pyplot as plt



def plot_fig(batches, start, stop ,step):
    y = np.arange(start, stop, step)
    file_prefix = "data/batch_"
    fig, ax = plt.subplots()
    ax.set_ylabel("config")
    ax.set_xlabel(r"$log_2 time$")
    ax.set_title("Model inference time on M1 CPU")
    for batch in batches:
        with open(file_prefix + str(batch) + "/cpu_profiler_res" + ".txt") as f:
            lines = f.readlines()
            count = 1
            x = []
            for line in lines:
                if count % 3 == 0:
                    x.append(math.log2(float(re.findall(r"\d+\.?\d*", line)[-1])/1000))
                count += 1
        ax.plot(x, y, label="batch=" + str(batch))
        ax.legend(loc = 'lower right', bbox_to_anchor=(0.73, 0.0))

    plt.savefig("inference_time_on_M1_CPU.png")   

if __name__ == "__main__":
    file_prefix = "data/batch_"
    batch_size = [1, 2, 4, 8, 16, 32]
    plot_fig(batch_size, 0.1, 1.01, 0.1)

