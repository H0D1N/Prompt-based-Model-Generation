import torch
import time
from model.utils import *

warmups = 2 # warmup number before sampling
runs = 3    # actual profiling number in each sampling

batch_size = [1]
modules = 18
sample_num = 2000    # number of samples in each batch size
config_indices, configs = configs_random(sample_num)
overall_latency = [0] * sample_num

if __name__ == "__main__":
    for batch in batch_size:
        inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float)

        for config_num, config in enumerate(configs):
            model = get_model(config)
            model.eval()

            for _ in range(warmups):
                model(inputs)

            for _ in range(runs):
                start_time = time.time()
                model(inputs)
                end_time = time.time()
                print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Batch size: " + str(batch), "    Sample No:", config_num)
                print("Current config:", config)
                print("Inference time:", (end_time - start_time) * 1000, "ms")
                overall_latency[config_num] += (end_time - start_time) * 1000
            overall_latency[config_num] /= runs

        with open("data/latency_time_new.txt", "w") as f:
            for count, config in enumerate(configs):
                for confg_data in config:
                    f.write(str(confg_data) + " ")
                f.write("\t")
                f.write(str(overall_latency[count]) + "\n")