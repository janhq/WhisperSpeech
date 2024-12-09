WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "/root/WhisperSpeech/ichigo-quantizer/checkpoints/loss=5.47.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "large-v3-vi-2d-2048c-dim64" \
    --whisper-name "large-v3" \
    --language "vi" \
    --batch-size 80
