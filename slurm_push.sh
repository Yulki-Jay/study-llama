#!/bin/bash
#SBATCH --job-name=bert-finetune           # 作业名称
#SBATCH --output=/home/jiangyunqi/study/study-llama/logs/stufy-llama-finetune-peft_v1.txt        # 输出日志的文件名
#SBATCH --mem=0                   # 任务不限制使用内存
#SBATCH --partition=gpujl          # 队列名称为gpujl


#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0

echo "4张显卡跑代码"
source /home/jiangyunqi/anaconda3/bin/activate study-llama-v2-finetune      # 激活conda环境
gpustat 
srun python3 /home/jiangyunqi/study/study-llama/bert-finetune/bert_finetune.py

echo "这个工作完成啦"