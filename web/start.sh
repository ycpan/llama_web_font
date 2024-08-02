#!/bin/bash
PYTHON=""
# python程序位置，可搭配一键包或是省去每次切换环境

#while true
#do
#    if [ -z "$PYTHON" ]; then
#        CUDA_VISIBLE_DEVICES=1 python wenda.py -t llama
#    else
#        CUDA_VISIBLE_DEVICES=1 $PYTHON wenda.py -t llama
#    fi
#sleep 1
#done
source ~/miniconda3/bin/activate llama_web_font 
CUDA_VISIBLE_DEVICES=0 nohup python wenda.py -t zhixing  >log.log 2>error.log &
#python wenda.py -t llama
#python wenda.py -t llama
