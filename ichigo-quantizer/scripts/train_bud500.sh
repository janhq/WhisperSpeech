CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
    --task "vq_stoks large-v3-vi-2d-512c-dim64" \
    --batch-size 16 \
    --iterations 100000 \
    --validate-every-n-steps 10000 \
    --training-data "linhtran92/viet_bud500 --language vi" \
    --validation-data "linhtran92/viet_bud500 --language vi --validation" \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "whisper-vq-vietnamese" \
    --num-gpus 1 \
