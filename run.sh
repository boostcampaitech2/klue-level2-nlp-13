#!/bin/bash

# python train.py --max_length 256 --batch_size 96 --epochs 5 --model_name klue/roberta-base --run_name entity_mark_punc
# python train.py --max_length 200 --batch_size 32 --epochs 7 --model_name klue/roberta-large --run_name entity_mark_punc_large


# python train.py --max_length 200 --batch_size 32 --epochs 7 --model_name klue/roberta-large --run_name entity_mark_punc_large_random --train_entity_set entity_marker_punct_random --dev_entity_set entity_marker_punct
# python train.py --max_length 200 --batch_size 32 --epochs 7 --model_name klue/roberta-large --run_name entity_mark_punc_large_random_except --train_entity_set entity_marker_punct_random_except --dev_entity_set entity_marker_punct

# python train.py --max_length 200 --batch_size 96 --epochs 3 --model_name klue/roberta-base --run_name entity_mark_punc_base_random_argu --train_entity_set entity_marker_punct --dev_entity_set entity_marker_punct
python train.py --max_length 200 --batch_size 32 --epochs 6 --model_name klue/roberta-large --run_name entity_mark_punc_large_argu --train_entity_set entity_marker_punct --dev_entity_set entity_marker_punct

# 최종
# python train.py --max_length 200 --batch_size 32 --epochs 6 --model_name klue/roberta-large --run_name entity_mark_punc_large_argu --train_entity_set entity_marker_punct --dev_entity_set entity_marker_punct


# python train.py --max_length 200 --batch_size 256 --epochs 1 --model_name klue/roberta-small --run_name entity_mark_punc_small_argu_random --train_entity_set entity_marker_punct --dev_entity_set entity_marker_punct
# python train.py --max_length 200 --batch_size 256 --epochs 1 --model_name klue/roberta-small --run_name entity_mark_punc_base_random_except --train_entity_set entity_marker_punct_random_except --dev_entity_set entity_marker_punct