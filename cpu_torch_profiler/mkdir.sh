#!/bin/zsh

cd "data_4_modling/batch_1"
# 声明和初始化数组

count = 18
# 创建文件夹
for ((i=1; i<=18; i++))
do
  folder_name="module_$i.txt"
  touch $folder_name
  echo "已创建文件 $folder_name"
  # cd $folder_name
  # # 循环输出数组中的数字
  #   for number in "${numbers[@]}"
  #   do
  #       mkdir "batch_$number"
  #   done
  #   cd ..
done


