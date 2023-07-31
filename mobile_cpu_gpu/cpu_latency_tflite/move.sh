#!/bin/bash

for ((i=1001; i<=2000; i++))
do
    file="data/results/${i}.txt"

    if [ -f "$file" ]; then
        rm "$file"
        echo "Deleted file: $file"
    else
        echo "File not found: $file"
    fi
done
