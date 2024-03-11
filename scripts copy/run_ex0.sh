#!/bin/bash
python src/main.py \
    configs/conceptual_captions/conceptual_captions.jsonnet \
    --mode train \
    --experiment_name finetune_T0_3B_mapping_network_on_conceptual_captions \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=1 \
    train.batch_size=4 \
    valid.step_size=1 \
    valid.batch_size=4 \
    train.additional.gradient_accumulation_steps=2 \
    train.lr=0.0003