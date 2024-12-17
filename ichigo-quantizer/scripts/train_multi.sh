WANDB_ENTITY="janai" python -m scripts.train \
    --num-gpus 8 \
    --task "vq_stoks medium-vi-2d-2048c-dim64" \
    --batch-size 42 \
    --epochs 100 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --run-name "p1 - vivoice+librittsr - w1090" \
    --load-checkpoint "checkpoints/whisper-vq-stoks-v3-7lang.model"
# --resume-from "checkpoints/vq_stoks/vq_stoks-init ckpt 2048 - 512 dup_noise - bs42 - max_token200 - mix_data_73 - wo_w_loss - 100e - ddp8-step=16248-val/loss=2.97.ckpt"

# # Phase 1 (w/ KL)
# WANDB_ENTITY="janai" python -m scripts.train \
#     --num-gpus 8 \
#     --task "vq_stoks medium-vi-2d-2048c-dim64" \
#     --batch-size 42 \
#     --epochs 100 \
#     --tunables "--rope --mask_embs --downsample_mean" \
#     --wandb-task-name "ichigo-quantizer" \
#     --run-name "p1 - vivoice+librittsr - w1090" \
#     --load-checkpoint "checkpoints/whisper-vq-stoks-v3-7lang.model" \
#     --phase 1

# # Phase 2 (w/o KL)
# WANDB_ENTITY="janai" python -m scripts.train \
#     --num-gpus 8 \
#     --task "vq_stoks medium-vi-2d-2048c-dim64" \
#     --batch-size 42 \
#     --epochs 100 \
#     --tunables "--rope --mask_embs --downsample_mean" \
#     --wandb-task-name "ichigo-quantizer" \
#     --run-name "p2 - vivoice+librittsr - w1090" \
#     --resume-from "path/to/phase1/checkpoint.ckpt" \
#     --phase 2
