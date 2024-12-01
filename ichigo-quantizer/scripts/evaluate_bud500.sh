CUDA_VISIBLE_DEVICES=2 python -m scripts.test \ 
--model-path "/root/ichigo-quantizer/checkpoints/vq_stoks-jennifer_thistle-step=92261-val_loss=4.41.ckpt" \
    --test-data "linhtran92/viet_bud500" \
    --model-size "large-v3-vi-2d-512c-dim64" \
    --language "vi" \
    --batch-size 8
