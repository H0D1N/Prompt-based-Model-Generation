#!/bin/bash
python torch2onnx.py
export device="417d6d91"    # 小米12在linux下的adb设备号
export device_dir="/data/local/tmp/pbmg"
export local_dir="/home/sunyi/Documents/AIOT/PBMG/Prompt-based-Model-Generation/mobile_cpu_gpu/latency_ort"

adb -s $device shell "mkdir -p $device_dir/models $device_dir/libs $device_dir/results"
adb -s $device push --sync $local_dir/libonnxruntime.so $device_dir/libs
adb -s $device push --sync $local_dir/ort_benchmark $device_dir

export smaple_num=1000
# iterate 500 timeto push model, run and pull result
for ((i=1; i<=$smaple_num; i ++))
do
    adb -s $device push --sync $local_dir/onnx_models/$i.onnx $device_dir/models
    adb -s $device shell "touch $device_dir/results/$i.txt"
    adb -s $device shell "export LD_LIBRARY_PATH=$device_dir/libs && .$device_dir/ort_benchmark --graph=$device_dir/models/$i.onnx --backend=arm --nums_warmup=5 --num_runs=5 --num_threads=4 --enable_op_profiling=true --prefix=$device_dir/results/$i > $device_dir/results/$i.txt " 
    adb -s $device pull --sync $device_dir/results/ $local_dir/data/
    adb -s $device shell "rm -rf $device_dir/models/*"
    adb -s $device shell "rm -rf $device_dir/results/*"
done