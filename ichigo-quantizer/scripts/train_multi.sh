# Bud500: 634158 train, 7500 val, 7500 test                                                                                                                                                                         ─╯
WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m scripts.train \
    --task "vq_stoks medium-vi-2d-1024c-dim64" \
    --batch-size 80 \
    --iterations 80000 \
    --validate-every-n-steps 1982 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --run-name "init ckpt 1024 - 512 random - bs80 - ddp - max_token20 - 10e" \
    --load-checkpoint "/root/WhisperSpeech/ichigo-quantizer/checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --num-gpus 4
