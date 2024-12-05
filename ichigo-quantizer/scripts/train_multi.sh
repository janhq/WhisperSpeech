WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m scripts.train \
    --task "vq_stoks medium-vi-2d-1024c-dim64" \
    --batch-size 80 \
    --iterations 25000 \
    --validate-every-n-steps 2500 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --run-name "init ckpt 1024 - 512 random - bs80 - max_token20 - mix_data" \
    --load-checkpoint "/root/WhisperSpeech/ichigo-quantizer/checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --num-gpus 4
