dataset="seq-cifar100"
SAVE_FOLDER="data/logfile/resnet18/experiment"
LOG_NAME="hisc"
gpu_id=(0)
lx_list=(0.001)
seed=(12345)
buffer_coeff=1.0
curr_coeff=-1.0
layer_list=("0_1_2_3_4")
interact_hsic=(-0.75)
buffer_size=500


i=0
    id=$((${gpu_id[$i]} % 4))
    echo "Using GPU ${id}"
    echo "interact hsic is ${interact_hsic[$i]}"
    python utils/main.py \
        --dataset=${dataset} \
        --model=derpp_hsic \
        --buffer_size=${buffer_size} \
        --lr=0.03 \
        --alpha=0.1 \
        --beta=0.5 \
        --optim=sgd \
        --log-filename ${SAVE_FOLDER}/paper/${dataset}/buffer_${buffer_size}_seed_${seed[i]}_${LOG_NAME}_${layer_list}_lx_${lx_list}_siam_${interact_hsic[$i]}.txt \
        --lambda_x=${lx_list} \
        --lambda_y=0.05 \
        --seed=${seed[i]} \
        --interact_hsic=${interact_hsic[$i]} \
        --buffer_hsic=${buffer_coeff} \
        --current_hsic=${curr_coeff} \
        --use_cl_mask \
        --use_siam \
        --hsic_features_to_include=${layer_list}\
        --gpu_id=${gpu_id[$i]} &
