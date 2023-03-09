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
seq_len=721
model_name=patchTST


model_id_name=ETTh2
data_name=ETTh2

random_seed=2021
for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --model_id patchtst'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --batch_size 128 --learning_rate 0.0001
done