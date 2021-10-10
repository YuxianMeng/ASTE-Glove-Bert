import os
import sys

# res14
os.system('CUDA_VISIBLE_DEVICES=0   python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 200 --batch_size 8 --if_lambda 0')
# os.system('CUDA_VISIBLE_DEVICES=0   python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 100 --batch_size 32 --if_lambda 0')
os.system('CUDA_VISIBLE_DEVICES=0   python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 200 --batch_size 32 --if_lambda 0')
# os.system('CUDA_VISIBLE_DEVICES=0   python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 100 --batch_size 8 --if_lambda 0')