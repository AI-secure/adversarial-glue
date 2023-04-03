TASK_NAME=rte

python -m paddle.distributed.launch \
run_glue_distributed.py \
    --task $TASK_NAME \
    --data_dir ./data/glue/$TASK_NAME/ \
    --bsz 16 \
    --epoch 5 \
    --lr 3e-5 \
    --from_pretrained ernie-2.0-large-en \
    --save_dir ./models/large/$TASK_NAME/
