#!/bin/zsh

cd "modling_data"
# 声明和初始化数组
numbers=(1 8 32)

count = 18
# 创建文件夹
for ((i=1; i<=18; i++))
do
  folder_name="module_$i"
  mkdir $folder_name
  echo "已创建文件夹 $folder_name"
  cd $folder_name
  # 循环输出数组中的数字
    for number in "${numbers[@]}"
    do
        mkdir "batch_$number"
    done
    cd ..
done


