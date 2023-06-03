# Number Game Task

## Introduction
In this task, we concatenate four images from the MINIST dataset together to form our training images. We have defined a set of tasks, including "How many number 1s are here?" and "Are there two number 2s here?" There are 50 tasks for training and 9 tasks for testing (but you can adjust this as needed). We treat this task as a binary classification problem, where images that meet the task's question are labeled as "Yes," and those that don't are labeled as "No."

## Training

Please use the following commands for training：  
~~~
python main.py <MINISt_Dataset_Path> --ada_kernel --task_loc <your task file> --epochs <epoch_num> -b <batch_size>
~~~
The training task file I'm using has been placed inside the 'task' folder

## Testing

Please use the following commands for testing：
~~~
python test.py <MINIST_Dataset_Path>  --ada_kernel --prompt <choose a specific task> --usage <choose a gate usage constraint example:40>  (--task_loc <your task file>)
~~~


