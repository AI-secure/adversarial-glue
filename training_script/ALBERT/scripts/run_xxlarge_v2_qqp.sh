TASK_NAME=qqp
lr=5e-5
batch_size=2
gradient_accumulation_steps=64
dr=0.1
max_steps=14000
warmup=1000

python run_glue.py \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --max_steps ${max_steps} \
  --output_dir ./models/xxlarge/$TASK_NAME/ \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir ./models/xxlarge/$TASK_NAME/ \
  --save_total_limit 1 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob ${dr}
