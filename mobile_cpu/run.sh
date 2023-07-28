#!/bin/bash
export device="417d6d91"
export device_dir="/data/local/tmp/hcp"

adb -s $device shell "mkdir -p $device_dir/models $device_dir/libs $device_dir/results"
adb -s $device push --sync /home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_ort_profiling/libonnxruntime.so $device_dir/libs
adb -s $device push --sync /home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_ort_profiling/onnx_models/config_0.1_model.onnx $device_dir/models
adb -s $device push --sync /home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_ort_profiling/ort_benchmark $device_dir

adb -s $device shell "export LD_LIBRARY_PATH=$device_dir/libs && .$device_dir/ort_benchmark \
--graph=$device_dir/models/config_0.1_model.onnx \
--backend=arm --nums_warmup=1 --num_runs=1 --num_threads=4 --enable_op_profiling=true --prefix=$device_dir/results/1"
# adb -s $device pull --sync $device_dir/results /home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_ort_profiling/tmp
# # adb -s $device pull --sync /data/local/tmp /home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_ort_profiling/tmpold