WANDB_ENTITY="janai" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m scripts.train \
    --task "vq_stoks medium-vi-2d-2048c-dim64" \
    --batch-size 42 \
    --iterations 24900 \
    --validate-every-n-steps 249 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --run-name "init ckpt 2048 - 512 dup_noise - bs42 - max_token200 - mix_data_73 - wo_w_loss - 100e - ddp8" \
    --load-checkpoint "/root/WhisperSpeech/ichigo-quantizer/checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --num-gpus 8
