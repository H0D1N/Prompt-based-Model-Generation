import torch
from model.utils import *
from torch.profiler import profile, ProfilerActivity

warmups = 10
runs = 10

batch_size = [1]
modules = 18
configs_num = 500
config_indices, configs = configs_random(configs_num)

modul_config = [[] for _ in range(modules)] # 记录每一次训练的config
module_data = [[] for _ in range(modules) ] # 记录每一次训练的module latency
overall_latency = []

for batch in batch_size:
    file_prefix = "data/cpu_modeling/"+ "/batch_" + str(batch) 
    inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float)

    for config_num, config in enumerate(configs):
        model = get_model(config)
        trace_name = file_prefix + "/raw_data/run_No." + str(config_num) + "_trace.json"
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Batch size: " + str(batch))
        print("Config num:", config_num)
        print("Current config:", config)

        def trace_handler(p):
            print(p.key_averages().table())
            print("Average inference time:", p.profiler.self_cpu_time_total/runs/1000, "ms")
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------- \n\n")
            p.export_chrome_trace(trace_name)        
        
        with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
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
        with open(file_prefix + "/module_" + str(module_num + 1) + ".txt", "w") as f:
            # 输入当前module对应的config的index
            for index in config_indices[module_num]:
                f.write(str(index) + "\t")
            f.write("\n")
            for count in range(configs_num):
                # 输入当前module每一次训练的config
                for config in modul_config[module_num][count]:
                    f.write(str(config) + "\t")
                f.write("\t")
                # 输入当前module每一次训练的latency
                for latency in module_data[module_num][count]:
                    f.write(str(latency) + "\t")
                f.write("\t")
                for total_latency in overall_latency[count]:
                    f.write(str(total_latency) + "\t")
                f.write("\n")