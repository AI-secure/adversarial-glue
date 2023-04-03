TASK_NAME=mnli

python -m paddle.distributed.launch \
run_glue_distributed.py \
    --task $TASK_NAME \
    --data_dir ./data/glue/$TASK_NAME/ \
    --bsz 512 \
    --epoch 3 \
    --lr 3e-5 \
    --from_pretrained ernie-2.0-en \
    --save_dir ./models/base/$TASK_NAME/
