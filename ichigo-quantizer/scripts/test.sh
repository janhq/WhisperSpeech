WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "checkpoints/epoch_accuracy=0.86387.ckpt" \
    --test-data "capleaf/viVoice" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --whisper-name "medium" \
    --language "vi" \
    --batch-size 1

WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=1 python -m scripts.test \
    --model-path "checkpoints/epoch_accuracy=0.86387.ckpt" \
    --test-data "parler-tts/libritts_r_filtered" \
    --model-size "medium-vi-2d-2048c-dim64" \
    --whisper-name "medium" \
    --language "en" \
    --batch-size 1
