export PYTHONPATH=/home/wurujie/workspace/code/vamba:$PYTHONPATH

task=videomme
run_name=vamba_pretrain_1e-5_image_mm_1258k_wh28_1077k_qkvo_nomamba_noalpha
checkpoints=(1000 2000 3000 4000 5000 6000 7000 8000 8412)
output_dir=res/${run_name}

for ckpt in "${checkpoints[@]}"; do
  echo "==== Evaluating checkpoint ${ckpt} on task ${task} ===="

  accelerate launch --num_processes 8 --main_process_port 52020 eval/eval_videomme.py \
    --model_name_or_path output/${run_name}/checkpoint-${ckpt} \
    --results_dir ${output_dir}

  echo "==== Completed checkpoint ${ckpt} ===="
done
