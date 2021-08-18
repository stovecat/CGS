dataset=$1
data_setting=$2
cmodel=$3
gmodel=$4

python train.py --num_of_epoch 100 --gpu --device 0 --dataset $dataset --data_setting $data_setting --data_augment --train_bert --cmodel $cmodel --gmodel $gmodel --imbalanced_ratio 100 --random_seed 7777
python train.py --num_of_epoch 100 --gpu --device 0 --dataset $dataset --data_setting $data_setting --data_augment --train_bert --cmodel $cmodel --gmodel $gmodel --imbalanced_ratio 100 --random_seed 7778
python train.py --num_of_epoch 100 --gpu --device 0 --dataset $dataset --data_setting $data_setting --data_augment --train_bert --cmodel $cmodel --gmodel $gmodel --imbalanced_ratio 100 --random_seed 7779
python train.py --num_of_epoch 100 --gpu --device 0 --dataset $dataset --data_setting $data_setting --data_augment --train_bert --cmodel $cmodel --gmodel $gmodel --imbalanced_ratio 100 --random_seed 7780
python train.py --num_of_epoch 100 --gpu --device 0 --dataset $dataset --data_setting $data_setting --data_augment --train_bert --cmodel $cmodel --gmodel $gmodel --imbalanced_ratio 100 --random_seed 7781
