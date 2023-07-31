#!/bin/bash
# python onnx2tflite.py
export device="417d6d91"    # 小米12在linux下的adb设备号
export device_dir="/data/local/tmp/pbmg"
export local_dir="/home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_gpu/gpu_latency_cpu_memory_tflite"

adb -s $device shell "mkdir -p $device_dir/models $device_dir/results"
adb -s $device push --sync $local_dir/android_aarch64_benchmark_model $device_dir

export smaple_num=1000
# iterate 1000 times to push model, run and pull result
for ((i=1; i<=$smaple_num; i ++))
do
    adb -s $device push --sync $local_dir/../tflite_models/$i.tflite $device_dir/models
    adb -s $device shell "touch $device_dir/results/$i.txt"
    adb -s $device shell "chmod +x $device_dir/android_aarch64_benchmark_model"
    adb -s $device shell ".$device_dir/android_aarch64_benchmark_model \
    --graph=$device_dir/models/$i.tflite  --warmup_runs=5 --num_runs=5 \
    --num_threads=4  --enable_op_profiling=true 
    --profiling_output_csv_file=$device_dir/results/$i.csv  > $device_dir/results/$i.txt" 
    # --use_gpu=true
    # --report_peak_memory_footprint=true --memory_footprint_check_interval_ms=10 \
    adb -s $device pull --sync $device_dir/results/ $local_dir/data/
    adb -s $device shell "rm -rf $device_dir/models/*"
    adb -s $device shell "rm -rf $device_dir/results/*"
done