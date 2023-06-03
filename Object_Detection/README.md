# Object Detection Task

## Introduction
For the object detection task, we have modified the COCO dataset by dividing each class into finer-grained categories. We used a VQA model to classify objects based on their colors and, following the recommendations from GPT, we merged colors to obtain six color categories for each object class. These color categories serve as our task set. The code for this task is based on modifications to the efficientdet code, so you can refer to the structure of the efficientdet code to understand our code. Our task is a single-class object detection task, so the accuracy may be slightly lower compared to multi-class object detection.

## Training
Train using the dataset I have on Alibaba Cloud's laboratory, located at "/mnt/data/airaiot/xuwenxing/efficent/mscoco," and modify the task.json file according to your requirements. In task.json, the last element of each task represents the category to be recognized, while the preceding elements represent the colors to be recognized for that task. '-1' denotes any color.   
You can start your training using the following command:
~~~
./distributed_train.sh <GPU_nums>  <Dataset_Path> --model resdet50 -b <batch_size> --amp  --lr <learning_rate> --warmup-epochs <epoch_num for warmup>  --sync-bn  --dataset coco2017 --momentum 0.9 --num-classes 1 --epochs <epoch_num> 
~~~
Please adjust your training command flexibly according to your specific needs.