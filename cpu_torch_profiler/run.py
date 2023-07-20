import torch
from model.utils import *
from torch.profiler import profile, ProfilerActivity

length = 38
start= 0.1
stop = 1.01
step = 0.1
warmups = 2
runs = 3

batch_size = [1, 8, 32]
module_num = 18

for mn in range(module_num):
    config_index, configs = configs_generate(mn + 1)
    for batch in batch_size:
        # 针对每一个batch，创建一个txt文件，用于存储当前module的所有config的运行时间，即全部的训练数据
        file = open("modling_data/module_"+ str(mn + 1) + "/batch_" + str(batch) + "/training_data.txt", "w")
        # 写入当前module对应的config的位置
        for index in config_index:
            file.write(str(index) + "\t")
        file.write("\n")
        file.flush()

        # 每一个config对应上述txt文件中的一行，共runs个数据
        for config in configs:
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
            trace_name = "modling_data/module_"+ str(mn + 1) + "/batch_" + str(batch) + "/config_" + trace_name2 + "on_"  + "position_"+ trace_name1 + "trace.json"
            
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
            module_latency = get_module_latency(trace_name, runs, mn)
            # 将当前config下的运行时间写入txt文件中
            for latency in module_latency:
                file.write(str(latency) + "\t")
            file.write("\n")
            file.flush()

        file.close()