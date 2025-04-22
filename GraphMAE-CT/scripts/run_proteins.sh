dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="PROTEINS"
[ -z "${device}" ] && device=0

python main_graph.py \
    --device $device \
    --dataset $dataset \
    --mask_rate 0.25 \
    --aug_ratio 0.2 \
    --replace_rate 0 \
    --encoder "gin" \
    --decoder "gin" \
    --gnn_dropout 0.2 \
    --attn_drop 0.1 \
    --num_hidden 256 \
    --random_walk_length 32 \
    --num_layers_enc_gnn 3 \
    --num_layers_enc_transformer 2 \
    --num_layers_dec 1 \
    --num_trans_heads 4 \
    --max_epoch 100 \
    --max_epoch_f 0 \
    --lr 0.00015 \
    --weight_decay 0.0 \
    --activation prelu \
    --optimizer adam \
    --drop_edge_rate 0.0 \
    --loss_fn "sce" \
    --seeds 0 1 2 3 4 \
    --linear_prob \
    --use_cfg \
    --batch_size 32 \
    --alpha_l 1 \
    --aug_type 2