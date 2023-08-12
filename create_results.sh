#!/usr/bin/env bash

run_for_dataset_and_backbone() {

    python main.py \
        --root "${ROOT}" \
        --prompt_root "${PROMPT_ROOT}" \
        --log_root "${LOG_ROOT}" \
        --dataset "${DATASET}" \
        --backbone "${BACKBONE}" \
        --no-follow_up \
        --source 'classname'

    python main.py \
        --root "${ROOT}" \
        --prompt_root "${PROMPT_ROOT}" \
        --log_root "${LOG_ROOT}" \
        --dataset "${DATASET}" \
        --backbone "${BACKBONE}" \
        --no-follow_up \
        --source 'single_template'

    python main.py \
        --root "${ROOT}" \
        --prompt_root "${PROMPT_ROOT}" \
        --log_root "${LOG_ROOT}" \
        --dataset "${DATASET}" \
        --backbone "${BACKBONE}" \
        --no-follow_up \
        --source 'template_set'

    # FUDD results with K=|C|
    python main.py \
        --root "${ROOT}" \
        --prompt_root "${PROMPT_ROOT}" \
        --log_root "${LOG_ROOT}" \
        --dataset "${DATASET}" \
        --backbone "${BACKBONE}" \
        --no-follow_up \
        --source 'gpt'

    # FUDD results with K=10
    python main.py \
        --root "${ROOT}" \
        --prompt_root "${PROMPT_ROOT}" \
        --log_root "${LOG_ROOT}" \
        --dataset "${DATASET}" \
        --backbone "${BACKBONE}" \
        --follow_up \
        --source 'gpt'
}

run_both_backbones_for_dataset() {

    BACKBONE='ViT-B/32'
    run_for_dataset_and_backbone
    BACKBONE='ViT-L/14@336px'
    run_for_dataset_and_backbone

}

ROOT=''
PROMPT_ROOT='./differential_descriptions'
LOG_ROOT='./fudd_log_files'

#

DATASET='cub'
run_both_backbones_for_dataset

#

DATASET='dtd'
run_both_backbones_for_dataset

#

DATASET='eurosat'
run_both_backbones_for_dataset

#

DATASET='fgvc_aircraft'
run_both_backbones_for_dataset

#

DATASET='flowers'
run_both_backbones_for_dataset

#

DATASET='food101'
run_both_backbones_for_dataset

#

DATASET='imagenet'
run_both_backbones_for_dataset

#

DATASET='imagenet_v2'
run_both_backbones_for_dataset

#

DATASET='pets'
run_both_backbones_for_dataset

#

DATASET='places365'
run_both_backbones_for_dataset

#

DATASET='stanford_cars'
run_both_backbones_for_dataset

#

DATASET='stanford_dogs'
run_both_backbones_for_dataset
