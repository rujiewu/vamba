export PYTHONPATH=$(pwd)

model_name_or_path="Qwen/Qwen2-VL-7B-Instruct"
data_config_file="train/data_configs/llavaov_image.yaml"
trainable_modules="model"

learning_rate=1e-5
dataset="image_mm_1258k_wh28_1077k"
run_name="vamba_pretrain_${learning_rate}_${dataset}_qkvo_nomamba_noalpha"
output_dir="output/${run_name}"

export WANDB_ENTITY="rujiewu-bigai"
export WANDB_PROJECT="vamba"

wandb login ${WANDB_API_KEY}
wandb online

NODE_RANK=${NODE_RANK:-1}
MASTER_ADDR=${MASTER_ADDR:-"hgx-hyperplane01"}
MASTER_PORT=${MASTER_PORT:-52020}

config_file="train/accelerate_configs/accelerate_config_zero3.yaml"

accelerate launch --config_file=$config_file --num_machines=2 --num_processes=16 --machine_rank=$NODE_RANK --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT train/train_vamba.py \
    --model_name_or_path $model_name_or_path \
    --data_config_file $data_config_file \
    --run_name $run_name \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate $learning_rate \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --do_train True \
    --trainable_modules $trainable_modules \
    --max_img_seq_len 120000 \
    --max_txt_seq_len 124000 \
    --init_flex_attn_weights_from_self_attn \
    --gradient_checkpointing_kwargs '{"use_reentrant":"False"}' \