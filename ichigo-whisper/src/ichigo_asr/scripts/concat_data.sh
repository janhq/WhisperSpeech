#!/bin/bash

# Format: "dataset_path:language:hub_path:output_name"
datasets=(
    "linhtran92/viet_bud500:vi:jan-hq/bud500:viet_bud500"
    "parler-tts/libritts_r_filtered:en:jan-hq/libritts_r_filtered:libritts"
)

for dataset_config in "${datasets[@]}"; do
    IFS=':' read -r dataset_path language hub_path output_name <<<"$dataset_config"

    output_path="concatenated_dataset/${output_name}"

    echo "Processing dataset: $dataset_path"
    echo "Language: $language"
    echo "Output path: $output_path"
    echo "Hub path: $hub_path"
    echo "-------------------"

    python data/concat_dataset.py \
        --dataset_path "$dataset_path" \
        --output_path "$output_path" \
        --language "$language" \
        --push_to_hub "$hub_path"
done
