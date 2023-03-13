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

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1440
model_name=PatchTST


model_id_name=patchTST
data_name=patchTST

random_seed=2021

python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --model_id patchtst'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --seq_len 1440 \
      --enc_in 3 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 60\
      --stride 60\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --batch_size 32 --learning_rate 0.0001\
      --channel_in 3\
      --class 2