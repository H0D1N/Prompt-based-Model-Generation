import torch
from model.utils import *
import numpy as np
warmups = 20 # warmup number before sampling
runs = 10    # actual profiling number in each sampling

batch_size = [1, 2, 4]
modules = 18
sample_num = 500   # number of samples in each batch size
config_indices, configs = configs_random(sample_num)

device = torch.device("cuda")


for batch in batch_size:
    file_path = "data/cuda_modeling/"+ "/batch_" + str(batch) + ".txt"
    inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float).to(device)
    model_latency = np.zeros((sample_num))

    for config_num, config in enumerate(configs):
        timings = np.zeros((runs, 1))
        model = get_model(config).to(device)
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Batch size: " + str(batch), "    Sample No:", config_num)
        print("Current config:", config)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # GPU-WARM-UP
        for _ in range(warmups):
            _ = model(inputs)

        # SYNCHRONIZE
        torch.cuda.synchronize()

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
        print("latency acc: ", mean_syn, "ms", "    std: ", std_syn, "ms\n")
        model_latency[config_num] = mean_syn

    with open(file_path, 'w') as f:
        for count, config in enumerate(configs):
            for config_data in config:
                f.write(str(config_data) + " ")
            f.write("\t" + str(model_latency[count]) + "\n")