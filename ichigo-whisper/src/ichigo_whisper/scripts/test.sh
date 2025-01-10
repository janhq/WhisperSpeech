# viVoice w/ merge base codebook
WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
    --model-path "checkpoints/p2_5e.ckpt" \
    --basevq-path "checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --test-data "capleaf/viVoice" \
    --model-size "merge-medium-vi-2d-2560c-dim64" \
    --whisper-name "medium" \
    --language "vi" \
    --batch-size 1 \
    --num-samples 1000

# LibriTTS-R w/ merge base codebook
WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=1 python -m scripts.test \
    --model-path "checkpoints/p2_5e.ckpt" \
    --basevq-path "checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --test-data "parler-tts/libritts_r_filtered" \
    --model-size "merge-medium-vi-2d-2560c-dim64" \
    --whisper-name "medium" \
    --language "en" \
    --batch-size 1 \
    --num-samples 1000
