WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "checkpoints/loss=0.19.ckpt" \
    --test-data "parler-tts/libritts_r_filtered" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --whisper-name "medium" \
    --language "en" \
    --batch-size 80
