import os
import sys

# # lap14
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')

# # res15
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')

# # res16
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')

# # res14
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')
# os.system('CUDA_VISIBLE_DEVICES=1   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi')

# os.system('CUDA_VISIBLE_DEVICES=2   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')
# os.system('CUDA_VISIBLE_DEVICES=2   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')
# os.system('CUDA_VISIBLE_DEVICES=2   python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts1 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts2 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_in_two_stage.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts3 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --batch_size 32')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --batch_size 16')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --batch_size 8')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-4 --dataset res14 --num_epoch 100 --batch_size 8')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-4 --dataset res14 --num_epoch 100 --batch_size 32')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-4 --dataset res14 --num_epoch 100 --batch_size 16')

# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-3 --dataset res14 --num_epoch 100 --batch_size 8')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-3 --dataset res14 --num_epoch 100 --batch_size 32')
# os.system('CUDA_VISIBLE_DEVICES=0 python train_with_bert.py --learning_rate 1e-3 --dataset res14 --num_epoch 100 --batch_size 16')

# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 100 --lambda0 1 --lambda1 1 --lambda2 1 --lambda3 1')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 100 --lambda0 1 --lambda1 1 --lambda2 1 --lambda3 0')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 100 --lambda0 0 --lambda1 0 --lambda2 1 --lambda3 1')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 100 --lambda0 0 --lambda1 0 --lambda2 1 --lambda3 0')

# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --lambda0 1 --lambda1 1 --lambda2 1 --lambda3 1')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --lambda0 1 --lambda1 1 --lambda2 1 --lambda3 0')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --lambda0 0 --lambda1 0 --lambda2 1 --lambda3 1')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 1e-5 --dataset res14 --num_epoch 100 --lambda0 0 --lambda1 0 --lambda2 1 --lambda3 0')

# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 80 --lambda0 1 --lambda1 1 --lambda2 1 --lambda3 1')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 80 --lambda0 1 --lambda1 1 --lambda2 1 --lambda3 0')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 80 --lambda0 0 --lambda1 0 --lambda2 1 --lambda3 1')
# os.system(' CUDA_VISIBLE_DEVICES=0  python train_with_bert.py --learning_rate 5e-5 --dataset res14 --num_epoch 80 --lambda0 0 --lambda1 0 --lambda2 1 --lambda3 0')


# res14
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 150  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res14 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')


# lap14
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 150  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset lap14 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')


# res15
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 150  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res15 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')


# res16
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 32  --model_name ts0 --num_epoch 150  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 8  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')

os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 50  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')
os.system('CUDA_VISIBLE_DEVICES=0  python train_in_two_stage3.py  --dataset res16 --save True --learning_rate 0.001 --batch_size 16  --model_name ts0 --num_epoch 100  --l2reg 0.00001 --emb_for_ao pair_shared_multi  --emb_for_ps shared_multi --write_results 0 --use_aspect_opinion_sequence_mask 0  --gcn_layers_in_graph0 1 --repeats 5')