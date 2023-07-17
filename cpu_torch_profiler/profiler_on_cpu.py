import torch
from utils import *
from torch.profiler import profile, ProfilerActivity

length = 32
start= 0.1
stop = 1.01
step = 0.1
configs = generate_configs(length, start, stop, step)
warmups = 10
runs = 50
batch_size = 1
inputs = torch.randn(batch_size, 3, 640, 640, dtype=torch.float)

file = open("profile_res/cpu_profiler_res.txt", "w")

for config in  configs:
    model = get_model(config)

    def trace_handler(p):
        print(p.key_averages().table())
        print("--------------------------------------------------------------------------------------------- \nCurrent config is [", round(config[0],2), ", ……,", round(config[0],2)," ]")
        print("Average time is ", p.profiler.self_cpu_time_total/runs/1000, "ms")
        file.write("--------------------------------------------------------------------------------------------- \nCurrent config is [ " + str(round(config[0],2)) + ", ……, " + str(round(config[0],2)) + " ]\n")
        file.write("Average time is " + str(p.profiler.self_cpu_time_total/runs/1000) + "ms\n")
        file.flush()
        p.export_chrome_trace("profile_res/" + "config_" + str(round(config[0], 2)) + "_profiler_trace"  + ".json")

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

file.close()

plot_fig("profile_res/cpu_profiler_res", start, stop, step)