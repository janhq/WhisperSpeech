# 634158 train, 7500 val, 7500 test
CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
    --task "vq_stoks large-v3-vi-2d-1024c-dim64" \
    --batch-size 48 \
    --iterations 15000 \
    --validate-every-n-steps 4999 \
    --training-data "linhtran92/viet_bud500 --language vi" \
    --validation-data "linhtran92/viet_bud500 --language vi --validation" \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --num-gpus 1
