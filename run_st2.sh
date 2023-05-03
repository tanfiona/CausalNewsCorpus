### Cause-Effect-Signal Span Detection
# We replicate the work by winners of the CNC 2022 Subtask 2 @ CASE, Team 1Cademy.
# Original Repository: https://github.com/Gzhang-umich/1CademyTeamOfCASE

### 01. Baseline
# Train & Test
CUDA_VISIBLE_DEVICES=7 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 10 \
  --num_warmup_steps 200 \
  --output_dir "outs/baseline" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 8 \
  --report_to wandb \
  --task_name ner \
  --do_train --do_test \
  --train_file "data/V2/train_subtask2_grouped.csv" \
  --validation_file "data/V2/dev_subtask2_grouped.csv" \
  --test_file "data/V2/test_subtask2_grouped.csv" \
  --weight_decay 0.005 \
  --use_best_model

# Test ONLY given a trained model
CUDA_VISIBLE_DEVICES=3 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --model_name_or_path albert-xxlarge-v2 \
  --load_checkpoint_for_test "/home/fiona/CausalNewsCorpus/outs/baseline/epoch_5/pytorch_model.bin" \
  --output_dir "outs/baseline" \
  --per_device_test_batch_size 32 \
  --report_to wandb \
  --task_name ner \
  --do_test \
  --test_file "data/V2/test_subtask2_grouped.csv"
  
### 02. Baseline+BSS
CUDA_VISIBLE_DEVICES=6 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 10 \
  --num_warmup_steps 200 \
  --output_dir "outs/baseline_BSS" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 8 \
  --report_to wandb \
  --task_name ner \
  --do_train --do_test \
  --train_file "data/V2/train_subtask2_grouped.csv" \
  --validation_file "data/V2/dev_subtask2_grouped.csv" \
  --test_file "data/V2/test_subtask2_grouped.csv" \
  --weight_decay 0.005 \
  --postprocessing_position_selector \
  --beam_search \
  --use_best_model

# Test ONLY given a trained model
CUDA_VISIBLE_DEVICES=4 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --model_name_or_path albert-xxlarge-v2 \
  --load_checkpoint_for_test "/home/fiona/CausalNewsCorpus/outs/baseline_BSS/epoch_4/pytorch_model.bin" \
  --output_dir "outs/baseline_BSS" \
  --per_device_test_batch_size 32 \
  --report_to wandb \
  --task_name ner \
  --do_test \
  --test_file "data/V2/test_subtask2_grouped.csv" \
  --postprocessing_position_selector \
  --beam_search

### 03. Baseline+BSS+SC
# Train & Test
CUDA_VISIBLE_DEVICES=7 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_signal_cls.py \
--model_name_or_path "bert-base-uncased" \
--output_dir "outs/signal_cls" \
--train_file "data/V2/train_subtask2_grouped.csv" \
--validation_file "data/V2/dev_subtask2_grouped.csv" \
--num_train_epochs=10

# Direct signal's tokenizer and model path using "signal_model_and_tokenizer_path"
CUDA_VISIBLE_DEVICES=5 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 10 \
  --num_warmup_steps 200 \
  --output_dir "outs/baseline_BSS_ES" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 8 \
  --report_to wandb \
  --task_name ner \
  --do_train --do_test \
  --train_file "data/V2/train_subtask2_grouped.csv" \
  --validation_file "data/V2/dev_subtask2_grouped.csv" \
  --test_file "data/V2/test_subtask2_grouped.csv" \
  --weight_decay 0.005 \
  --postprocessing_position_selector \
  --beam_search \
  --signal_classification \
  --pretrained_signal_detector \
  --signal_model_and_tokenizer_path "outs/signal_cls" \
  --use_best_model

# Test ONLY given a trained model
CUDA_VISIBLE_DEVICES=3 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --model_name_or_path albert-xxlarge-v2 \
  --load_checkpoint_for_test "/home/fiona/CausalNewsCorpus/outs/baseline_BSS_ES/epoch_4/pytorch_model.bin" \
  --output_dir "outs/baseline_BSS_ES" \
  --per_device_test_batch_size 32 \
  --postprocessing_position_selector \
  --beam_search \
  --signal_classification \
  --pretrained_signal_detector \
  --signal_model_and_tokenizer_path "outs/signal_cls" \
  --task_name ner \
  --do_test \
  --test_file "data/V2/test_subtask2_grouped.csv"

### 04. Baseline+BSS+SC+DA

# generate augments (optional)
# adjust number of augments using "NUM_RETURN_SEQ" within script
CUDA_VISIBLE_DEVICES=1 \
/home/fiona/anaconda3/envs/py310/bin/python3 src/data_aug_st2.py

CUDA_VISIBLE_DEVICES=6 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 10 \
  --num_warmup_steps 200 \
  --output_dir "outs/baseline_BSS_ES_DA_1" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 8 \
  --report_to wandb \
  --task_name ner \
  --do_train --do_test \
  --train_file "data/V2/train_subtask2_grouped.csv" \
  --validation_file "data/V2/dev_subtask2_grouped.csv" \
  --test_file "data/V2/test_subtask2_grouped.csv" \
  --weight_decay 0.005 \
  --postprocessing_position_selector \
  --beam_search \
  --signal_classification \
  --pretrained_signal_detector \
  --signal_model_and_tokenizer_path "outs/signal_cls" \
  --augmentation_file "data/V2/augmented_subtask2_1_train.csv" \
  --use_best_model