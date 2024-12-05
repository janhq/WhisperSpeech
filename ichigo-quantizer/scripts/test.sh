WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=2 python -m scripts.test \
    --model-path "/root/test/WhisperSpeech/ichigo-quantizer/checkpoints/vq_stoks/vq_stoks-init ckpt 2048 - 512 random - bs80 - max_token20 - mix_data-step=3035-val/loss=1.50.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --language "vi" \
    --batch-size 128
