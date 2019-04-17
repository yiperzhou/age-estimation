#!/bin/bash
#
#SBATCH -J MTL_AlexNet_model
#
#SBATCH --output=log/
#SBATCH --error=log/
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40960
#SBATCH --time=4-23:59:00
#SBATCH --partition=gpu

module load CUDA
source /home/opt/anaconda3/bin/activate dl

# 
python main.py --model MTL_AlexNet_model --dataset CVPR_16_ChaLearn --lr_rate 0.001 --epoch 40  --batch_size 16 \
--load_pretrained_model True --multitask_training_type "Train_Valid" --loss_weights [1,1,1,1]

python main.py --model MTL_AlexNet_model --dataset CVPR_16_ChaLearn --lr_rate 0.001 --epoch 40  --batch_size 16 \
--load_pretrained_model True --multitask_training_type "Train_Valid_2" --loss_weights [1,1,1,1]