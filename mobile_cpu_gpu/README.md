1. cpu_latency's configs are in onnx_models/configs.txt
2. cpu_memory's configs are in onnx_models_op12/configs.txt
3. gpu_latency's configs are in onnx_models_op12/configs.txt
4. warmups and runs are shown in txt files in data folder, all with 1000 samples
5. model format conversion is in done in cpu_latency_ort/model/utils.py
6. exract inference time and memory usage from txt files