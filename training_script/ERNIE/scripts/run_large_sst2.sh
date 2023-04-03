TASK_NAME=sst2

python -m paddle.distributed.launch \
run_glue_distributed.py \
    --task $TASK_NAME \
    --data_dir ./data/glue/$TASK_NAME/ \
    --bsz 64 \
    --epoch 4 \
    --lr 2e-5 \
    --from_pretrained ernie-2.0-large-en \
    --save_dir ./models/large/$TASK_NAME/
