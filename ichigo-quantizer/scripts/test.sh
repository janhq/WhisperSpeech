WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "/root/WhisperSpeech/ichigo-quantizer/checkpoints/loss=0.19.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --whisper-name "medium" \
    --language "vi" \
    --batch-size 100
