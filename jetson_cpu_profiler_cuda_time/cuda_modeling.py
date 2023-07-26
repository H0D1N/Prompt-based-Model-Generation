import torch
from model.utils import *
import numpy as np
print("dwwdwwdwdwd")
warmups = 10
runs = 10

batch_size = [1]
modules = 18
configs_num = 1
config_indices, configs = configs_random(configs_num)
print(configs)
timings = np.zeros((runs, 1))
device = torch.device("cuda")
for batch in batch_size:
    file_prefix = "data/cuda_modeling/"+ "/batch_" + str(batch)
    inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float).to(device)

    for config_num, config in enumerate(configs):
        model = get_model(config).to(device)
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Batch size: " + str(batch))
        print("Config num:", config_num)
        print("Current config:", config)

##################### use cuda.synchronize() to measure total latency #####################
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


        # GPU-WARM-UP
        for _ in range(warmups):
            _ = model(inputs)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(runs):
                starter.record()
                _ = model(inputs)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / runs
        std_syn = np.std(timings)
        print(mean_syn)
        print(std_syn)