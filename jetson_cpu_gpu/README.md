1. Jetson平台运行pytorch的docker容器，在容器内每次都需要先安装requirements.txt中的依赖包
2. Jetson CPU上采用torch.profiler模块测试，Jetson GPU上采用time.time()测试
3. extract inference time and memory usage from txt files
4. configs data can be inferred from txt files too.
5. cpu_latency data are recorded like server_cpu modules in module_?.txt files, run 10 times, sample 500, with warmups data not included.
6. cpu_memory data are recorded like server_cpu modules in module_?.txt files, run 10 times, sample 500, with warmups data not included.
7. gpu_latency data are recorded in a simpler way, run 10 times, sample 500, with warmups not included, configs data and avg latency in each line of txt files.