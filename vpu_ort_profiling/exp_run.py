from model.utils import *
import onnxruntime as ort
import numpy as np
warmups = 0 # warmup number before sampling
runs = 1    # actual profiling number in each sampling
batch_size = [1]
modules = 18
sample_num = 1    # number of samples in each batch size


config_indices, configs = configs_random(sample_num)
modul_config = [[] for _ in range(modules)] # configs of each module in each sample
module_data = [[] for _ in range(modules) ] # latency of each module in each sample
overall_latency = []

print("Device type selected is MYRIAD_FP16 VPU using the OpenVINO Execution Provider")
    
for batch in batch_size:
    file_prefix = "data/modeling/"+ "/batch_" + str(batch) 


    for config_num, config in enumerate(configs):
        # model = get_model(config)
        model_path = "config_0.4_model.onnx"

        trace_name = file_prefix + "/raw_data/run_No." + str(config_num) + "_trace.json"
        so = ort.SessionOptions()
        so.enable_profiling = True
        so.profile_file_prefix = trace_name
        # OpenVINO™ backend performs hardware, dependent as well as independent optimizations on the graph to infer it on the target hardware with best possible performance. In most cases it has been observed that passing the ONNX input graph as is without explicit optimizations would lead to best possible optimizations at kernel level by OpenVINO™. 
        # For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime for OpenVINO™ Execution Provider.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        session = ort.InferenceSession(model_path, so, providers=["OpenVINOExecutionProvider"], provider_options=[{"device_type": "MYRIAD_FP16"}],)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_data = np.array(np.random.random_sample((1, 3, 640, 640)), dtype=np.float32)
        ort_inputs = {input_name: input_data}
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Batch size: " + str(batch))
        print("Sample No:", config_num)
        print("Current config:", config)

        for _ in range(warmups + runs):
             _ = session.run(None, ort_inputs)
        session.end_profiling()


    #     module_latency, totlal_latency = get_module_latency(trace_name, runs)
    #     for module_num in range(modules):
    #         left = config_indices[module_num][0]
    #         right = config_indices[module_num][-1] + 1
    #         modul_config[module_num].append(config[left:right])
    #         module_data[module_num].append(module_latency[module_num])

    #     overall_latency.append(totlal_latency)

    # for module_num in range(modules):
    #     with open(file_prefix + "/module_" + str(module_num + 1) + ".txt", "w") as f:
    #         # index of configs of current module
    #         for index in config_indices[module_num]:
    #             f.write(str(index) + "\t")
    #         f.write("\n")
    #         for count in range(sample_num):
    #             # write configs of current module
    #             for config in modul_config[module_num][count]:
    #                 f.write(str(config) + "\t")
    #             f.write("\t")
    #             # write latency of current module
    #             for latency in module_data[module_num][count]:
    #                 f.write(str(latency) + "\t")
    #             f.write("\t")
    #             for total_latency in overall_latency[count]:
    #                 f.write(str(total_latency) + "\t")
    #             f.write("\n")