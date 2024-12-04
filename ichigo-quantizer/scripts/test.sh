WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "/root/vq_stoks-1204024/vq_stoks-jonathan_aquamarine-step=14998-val/loss=3.33.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "medium-vi-2d-1024c-dim64" \
    --language "vi" \
    --batch-size 64
