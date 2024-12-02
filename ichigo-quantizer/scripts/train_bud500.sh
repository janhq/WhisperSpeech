# 634158 train, 7500 val, 7500 test                                                                                                                                                                         ─╯
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m scripts.train \
    --task "vq_stoks medium-vi-2d-1024c-dim64" \
    --batch-size 192 \
    --iterations 3500 \
    --validate-every-n-steps 1749 \
    --training-data "linhtran92/viet_bud500 --language vi" \
    --validation-data "linhtran92/viet_bud500 --language vi --validation" \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --load-checkpoint "/root/WhisperSpeech/ichigo-quantizer/checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --num-gpus 4
