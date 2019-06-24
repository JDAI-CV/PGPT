#! /bin/bash
gpu=1
cd ./inference
echo "Using the GPU of ${gpu}"
CUDA_VISIBLE_DEVICES=${gpu} python -u inference.py 2>&1 | tee ../log_inference.txt
cd ../
