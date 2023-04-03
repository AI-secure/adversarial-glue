TASK_NAME=qqp
num_epochs=10
warmup=500
lr=2e-5
num_gpus=8
batch_size=8

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --num_train_epochs ${num_epochs} \
  --warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --output_dir ./models/large/$TASK_NAME/ \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir ./models/large/$TASK_NAME/ \
  --save_total_limit 1