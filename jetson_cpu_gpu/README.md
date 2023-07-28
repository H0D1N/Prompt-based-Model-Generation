1. Jetson平台运行pytorch的docker容器，在容器内每次都需要先安装requirements.txt中的依赖包
2. Jetson CPU上采用torch.profiler模块测试，Jetson GPU上采用time.time()测试
3. cpu_latency_modeling.py负责cpu上的延时建模，cpu_memory_modeling.py负责cpu上的内存建模
4. gpu_latency_modeling.py负责gpu上的延时建模