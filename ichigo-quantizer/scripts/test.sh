WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "/root/WhisperSpeech/ichigo-quantizer/checkpoints/vq_stoks/vq_stoks-init ckpt 1024 - 512 dup noise - bs80 - max_token20 - mix_data-step=21722-val/loss=1.37.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --language "vi" \
    --batch-size 64
