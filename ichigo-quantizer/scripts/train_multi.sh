WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m scripts.train \
    --task "vq_stoks medium-vi-2d-2048c-dim64" \
    --batch-size 64 \
    --iterations 23330 \
    --validate-every-n-steps 2333 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --run-name "init ckpt 2048 - 512 dup_noise - bs64 - max_token20 - mix_data - rm_weightl" \
    --load-checkpoint "/root/WhisperSpeech/ichigo-quantizer/checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --num-gpus 8
