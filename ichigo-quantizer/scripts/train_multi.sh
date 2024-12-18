WANDB_ENTITY="janai" python -m scripts.train \
    --num-gpus 8 \
    --task "vq_stoks medium-vi-2d-2048c-dim64" \
    --batch-size 42 \
    --epochs 100 \
    --tunables "--rope --mask_embs --downsample_mean" \
    --wandb-task-name "ichigo-quantizer" \
    --run-name "p1 - vivoice+librittsr - w1090" \
    --load-checkpoint "checkpoints/whisper-vq-stoks-v3-7lang.model" \
    --phase 1

# # Phase 1 (w/ KL)
# WANDB_ENTITY="janai" python -m scripts.train \
#     --num-gpus 8 \
#     --task "vq_stoks medium-vi-2d-2048c-dim64" \
#     --batch-size 42 \
#     --epochs 100 \
#     --tunables "--rope --mask_embs --downsample_mean" \
#     --wandb-task-name "ichigo-quantizer" \
#     --run-name "p1 - vivoice+librittsr" \
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
#     --run-name "p2 - vivoice+librittsr" \
#     --resume-from "path/to/phase1/checkpoint.ckpt" \
#     --phase 2

#V2 --load-checkpoint "checkpoints/whisper-vq-stoks-medium-en+pl.model"
#V3 --load-checkpoint "checkpoints/whisper-vq-stoks-v3-7lang.model"
