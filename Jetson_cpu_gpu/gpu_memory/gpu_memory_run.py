import torch
import torch
from model.utils import *
from torch.profiler import profile, ProfilerActivity

warmups = 10
runs = 5

batch_size = [1]
# modules = 18
configs_num = 500
config_indices, configs = configs_random(configs_num)
all_memory = []

# modul_config = [[] for _ in range(modules)]
# module_data = [[] for _ in range(modules) ]
# overall_latency = []

# GPU-WARM-UP
model = get_model(configs[0]).cuda()
inputs = torch.randn(1, 3, 640, 640, dtype=torch.float).cuda()
for _ in range(warmups):
    _ = model(inputs)
# SYNCHRONIZE
torch.cuda.synchronize()

for batch in batch_size:
    # file_prefix = "data"+ "/batch_" + str(batch) 

    for config_num, config in enumerate(configs):
        model = get_model(config).cuda()
        inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float).cuda()
        # trace_name = file_prefix + "/raw_data/run_No." + str(config_num) + "_trace.json"
        
        # def trace_handler(p):
        #     print(p.key_averages().table())
        #     print("Average inference time:", p.profiler.self_cpu_time_total/runs/1000, "ms")
        #     print("------------------------------------------------------------------------------------------------------------------------------------------------------------- \n\n")
        #     p.export_chrome_trace(trace_name)       
        
        # inputs.cuda()
        # model.cuda()
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Batch size: " + str(batch))
        print("Sample num:", config_num)
        print("Current config:", config)
        memory_per_config = [0] * runs
 
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     profile_memory=True,
        #     schedule=torch.profiler.schedule(
        #         wait=0,
        #         warmup=warmups,
        #         active=runs),
        #     on_trace_ready=trace_handler
        # ) as prof:
        #     for idx in range(warmups + runs):
        #         model(inputs)
        #         prof.step()
                
       
        # print("now cuda memory")
        
        for rep in range(runs):
            torch.cuda.reset_peak_memory_stats()
            model(inputs)
            max_mem = torch.cuda.max_memory_allocated()
            memory_per_config[rep] = max_mem/1024/1024
            # SYNCHRONIZE
            torch.cuda.synchronize()
        all_memory.append(sum(memory_per_config) / runs)
        print("mems: ", memory_per_config)
        print("mem avg: ", all_memory[-1], "\n")
        
        
        # print("Now CPU:")
        # model = model.to("cpu")
        # inputs = inputs.to("cpu")
        # with profile(
        #     activities=[ProfilerActivity.CPU],
        #     profile_memory=True,
        #     schedule=torch.profiler.schedule(
        #         wait=0,
        #         warmup=warmups,
        #         active=runs),
        #     on_trace_ready=trace_handler
        # ) as prof:
        #     for idx in range(warmups + runs):
        #         model(inputs)
        #         prof.step()
        
        
    #     module_latency = get_module_latency(trace_name, runs)
    #     for module_num in range(modules):
    #         left = config_indices[module_num][0]
    #         right = config_indices[module_num][-1] + 1
    #         modul_config[module_num].append(config[left:right])
    #         module_data[module_num].append(module_latency[module_num])

    #     # overall_latency.append(totlal_latency)

    # for module_num in range(modules):
    #     with open(file_prefix + "/latency/module_" + str(module_num + 1) + ".txt", "w") as f:

    #         for index in config_indices[module_num]:
    #             f.write(str(index) + "\t")
    #         f.write("\n")
    #         for count in range(configs_num):

    #             for config in modul_config[module_num][count]:
    #                 f.write(str(config) + "\t")
    #             f.write("\t")

    #             for latency in module_data[module_num][count]:
    #                 f.write(str(latency) + "\t")
    #             # f.write("\t")
    #             # # for total_latency in overall_latency[count]:
    #             # #     f.write(str(total_latency) + "\t")
    #             f.write("\n")

with open("data/batch_1/configs.txt", "w") as f:
    for count in range(configs_num):
        for config in configs[count]:
            f.write(str(config) + "\t")
        f.write("\n")
with open("data/batch_1/memory_data.txt", "w") as f:
    for count in range(configs_num):
        f.write(str(all_memory[count]) + "\n")

