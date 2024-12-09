WANDB_ENTITY="janai" python -m scripts.train \
    --task "vq_stoks medium-vi-2d-2048c-dim64" \
    --batch-size 42 \
    --epochs 100 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --max-tokens 200 \
    --run-name "init ckpt 2048 - 512 dup_noise - bs42 - max_token200 - mix_data_73 - 100e - ddp8" \
    --load-checkpoint "checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --num-gpus 8

# WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m scripts.train \
#     --task "vq_stoks large-v3-vi-2d-2048c-dim64" \
#     --batch-size 24 \
#     --iterations 43600 \
#     --validate-every-n-steps 436 \
#     --tunables "--rope --mask_embs --downsample_mean" \
#     --wandb-task-name "ichigo-quantizer" \
#     --run-name "scratch 2048 - largev3 - bs24 - max_token200 - mix_data_73 - 100e - ddp8" \
#     --num-gpus 8
