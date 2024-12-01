CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --task "vq_stoks large-v3-vi-2d-512c-dim64" \
    --batch-size 8 \
    --iterations 100000 \
    --validate-every-n-steps 10000 \
    --training-data "linhtran92/viet_bud500 --language vi" \
    --validation-data "linhtran92/viet_bud500 --language vi --validation" \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "whisper-vq-vietnamese"
