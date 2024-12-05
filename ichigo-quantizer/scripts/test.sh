WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=1 python -m scripts.test \
    --model-path "/root/test/WhisperSpeech/ichigo-quantizer/checkpoints/vq_stoks/vq_stoks-joshua_darkslateblue-step=5809-val/loss=5.56.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "medium-vi-2d-1024c-dim64" \
    --language "vi" \
    --batch-size 128
