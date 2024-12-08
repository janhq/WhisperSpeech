WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "/root/WhisperSpeech/ichigo-quantizer/checkpoints/vq_stoks/vq_stoks-init ckpt 2048 - 512 dup_noise - bs42 - max_token200 - mix_data_73 - wo_w_loss - 100e - ddp8-step=16248-val/loss=2.97.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --language "vi" \
    --batch-size 100
