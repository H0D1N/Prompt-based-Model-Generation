import torch
from model.utils import *
from torch.profiler import profile, ProfilerActivity

warmups = 2 # warmup number before sampling
runs = 3    # actual profiling number in each sampling

batch_size = [1]
modules = 18
sample_num = 1000   # number of samples in each batch size
config_indices, configs = configs_random(sample_num)

modul_config = [[] for _ in range(modules)] # configs of each module in each sample
module_data = [[] for _ in range(modules) ] # latency of each module in each sample
overall_latency = []

for batch in batch_size:
    file_prefix = "data/memory/"+ "batch_" + str(batch) 
    inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float).cuda()

    for config_num, config in enumerate(configs):
        model = get_model(config)
        model = model.cuda()
        trace_name = file_prefix + "/raw_data/run_No." + str(config_num) + "_trace.json"
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Batch size: " + str(batch), "    Sample No:", config_num)
        print("Current config:", config)

        def trace_handler(p):
            with open(file_prefix + "/raw_data/run_No." + str(config_num) + ".txt", "w") as f:
                print(p.key_averages().table(), file=f)
                print("Average inference time:", p.profiler.self_cpu_time_total/runs/1000, "ms", file=f)
            print("Average inference time:", p.profiler.self_cpu_time_total/runs/1000, "ms")
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------- \n\n")
            p.export_chrome_trace(trace_name)        
        
        with profile(
<<<<<<< HEAD:server_cpu/memory_run.py
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True,
=======
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
>>>>>>> f221a4849d5aaed2fba50610c0942a4d3ed9887d:Desktop_cpu/memory_run.py
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=warmups,
                active=runs),
            on_trace_ready=trace_handler
        ) as prof:
            for idx in range(warmups + runs):
                model(inputs)
                prof.step()

        module_latency, totlal_latency = get_module_latency(trace_name, runs)
        for module_num in range(modules):
            left = config_indices[module_num][0]
            right = config_indices[module_num][-1] + 1
            modul_config[module_num].append(config[left:right])
            module_data[module_num].append(module_latency[module_num])

        overall_latency.append(totlal_latency)

    for module_num in range(modules):
        with open(file_prefix + "/latency/" + "/module_" + str(module_num + 1) + ".txt", "w") as f:
            # index of configs of current module
            for index in config_indices[module_num]:
                f.write(str(index) + "\t")
            f.write("\n")
            for count in range(sample_num):
                # write configs of current module
                for config in modul_config[module_num][count]:
                    f.write(str(config) + "\t")
                f.write("\t")
                # write latency of current module
                for latency in module_data[module_num][count]:
                    f.write(str(latency) + "\t")
                f.write("\t")
                for total_latency in overall_latency[count]:
                    f.write(str(total_latency) + "\t")
                f.write("\n")