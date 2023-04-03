TASK_NAME=sst2
lr=1e-5
num_gpus=8
batch_size=4
dr=0
max_steps=20935
warmup=1256

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --max_steps ${max_steps} \
  --output_dir ./models/xxlarge/$TASK_NAME/ \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir ./models/xxlarge/$TASK_NAME/ \
  --save_total_limit 1 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob ${dr}
