#!/bin/bash
#SBATCH --job-name=llama-finetune           # 作业名称
#SBATCH --output=/home/jiangyunqi/study/LLM/study-llama/logs/stufy-llama-finetune-peft-precision_v1.txt        # 输出日志的文件名
#SBATCH --mem=0                   # 任务不限制使用内存
#SBATCH --partition=gpujl          # 队列名称为gpujl


#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=0

echo "这是一次跑量化的任务，使用了8bit量化，优化器是由16精度，记录一下显存的使用情况"
source /home/jiangyunqi/anaconda3/bin/activate study-llama-v2-finetune      # 激活conda环境
gpustat 
srun python3 /home/jiangyunqi/study/LLM/study-llama/llama-finetune/main.py

# srun python3 /home/jiangyunqi/study/LLM/study-llama/bert-finetune/bert_finetune.py

echo "这个工作完成啦"