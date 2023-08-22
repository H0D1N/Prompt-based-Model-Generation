1. for ort framework, cpu_latency's configs and models are in old_models/onnx_models (op_set = newest, 1000 in total), for tflite framework, thses files are in onnx_models/, 2000 in total
2. cpu_memory's, gpu_latency's and gpu_memory's configs are in onnx_models/configs.txt, 2000 in total
3. warmups and runs are shown in txt files in data folder, all with 2000 samples for tflite framework (now we use tflite for all exp)
4. model format conversion is in done in model_conversion/
5. configs are stored in configs.txt
6. exract inference time and memory usage from txt files
7. warm ups and runs are both 5 times