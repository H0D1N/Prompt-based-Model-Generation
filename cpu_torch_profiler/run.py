import torch
from model.utils import *
from torch.profiler import profile, ProfilerActivity

warmups = 2
runs = 3

batch_size = [1]
modules = 18
configs_num = 5000
config_indices, configs = configs_random(configs_num)

for batch in batch_size:
    # 针对每一个batch，创建一个txt文件，用于存储当前module的所有config的运行时间，即全部的训练数据
    file = open("data_4_modling/"+ "/batch_" + str(batch) + "/training_data.txt", "w")

    # 每一个config对应上述txt文件中的一行，共runs个数据
    for config_num, config in enumerate(configs):
        inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float)
        model = get_model(config)

        trace_name1 = ""
        trace_name2 = ""
        for tmp in config_index:
            trace_name1 += str(tmp) + "_"
            trace_name2 += str(round(config[tmp], 1)) + "_"
            # 将config的值写入txt文件中
            file.write(str(round(config[tmp], 1)) + "\t")
        file.write("\t")
        trace_name = "modling_data/raw_data/"+ str(num) + "trace.json"
        
        # 运行模型，将详细数据保存在trace文件中
        def trace_handler(p):
            print(p.key_averages().table())
            print("Batch size: " + str(batch))
            print("Current config:", config)
            print("Average inference time:", p.profiler.self_cpu_time_total/runs/1000, "ms")
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------- \n\n")
            p.export_chrome_trace(trace_name)        
        
        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=warmups,
                active=runs),
            on_trace_ready=trace_handler
        ) as prof:
            for idx in range(warmups + runs):
                model(inputs)
                prof.step()


        # 从trace文件中读取数据当前module在当前config下的运行时间
        module_latency, totlal_latency = get_module_latency(trace_name, runs)
        for module_num in range(modules):
            
        # 将当前config下的运行时间写入txt文件中
        for latency in module_latency:
            file.write(str(latency) + "\t")
        file.write("\n")
        file.flush()

    file.close()