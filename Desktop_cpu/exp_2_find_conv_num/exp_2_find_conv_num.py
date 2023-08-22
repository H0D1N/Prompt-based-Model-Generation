import torch
from model.utils import *
from torch.profiler import profile, ProfilerActivity

length = 38
start= 0.1
stop = 1.01
step = 0.1
warmups = 4
runs = 1
batch_size = 1


batch = 1
inputs = torch.randn(batch, 3, 640, 640, dtype=torch.float)
file = open("box_class_backbone_fpn_profiler_res.txt", "w")
config = [1] * 38
backbone, fpn, class_net, box_net, model = get_model(config)
backbone.eval()
fpn.eval()
class_net.eval()
box_net.eval()
model.eval()

def trace_handler(p):
    print(p.key_averages().table())
    print("--------------------------------------------------------------------------------------------- \nCurrent config is [", round(config[0],2), ", ……,", round(config[0],2)," ]")
    print("Under batch " + str(batch) + ", average time is ", p.profiler.self_cpu_time_total/runs/1000, "ms")
    file.write("--------------------------------------------------------------------------------------------- \nCurrent config is [ " + str(round(config[0],2)) + ", ……, " + str(round(config[0],2)) + " ]\n")
    file.write("Under batch " + str(batch) + ", average time is " + str(p.profiler.self_cpu_time_total/runs/1000) + "ms\n")
    file.flush()
    p.export_chrome_trace("box_class_backbone_fpn_profiler_trace"  + ".json")

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