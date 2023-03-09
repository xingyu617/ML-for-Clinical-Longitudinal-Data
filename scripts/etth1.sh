#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx1080ti:2

#SBATCH -t 2-00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=test_pipeline.txt

source ~/.bashrc
enable_modules

module load python/3.8
source ~/CUDA/bin/activate



cd /cluster/home/chenxin/thesis
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=721

model_name = Linear

python -u run_longExp.py \
  --is_training 1 \
  --model_id ETTh1_$seq_len'_'96 \
  --model Linear \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 #>logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log

