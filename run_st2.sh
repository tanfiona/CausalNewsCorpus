### Cause-Effect-Signal Span Detection
# We replicate the work by winners of the CNC 2022 Subtask 2 @ CASE, Team 1Cademy.
# Original Repository: https://github.com/Gzhang-umich/1CademyTeamOfCASE

# Baseline (Train & Test)
sudo CUDA_VISIBLE_DEVICES=0 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 20 \
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

# Baseline (Test ONLY given a trained model)
sudo CUDA_VISIBLE_DEVICES=3 \
/home/fiona/anaconda3/envs/py310/bin/python3 run_st2.py \
  --model_name_or_path albert-xxlarge-v2 \
  --load_checkpoint_for_test "/home/fiona/CausalNewsCorpus/outs/baseline/epoch_3/pytorch_model.bin" \
  --output_dir "outs/baseline" \
  --per_device_test_batch_size 32 \
  --report_to wandb \
  --task_name ner \
  --do_test \
  --test_file "data/V2/test_subtask2_grouped.csv"
  

### Cause-Effect Span Detection

## We used the UniCausal (https://github.com/tanfiona/UniCausal) repository to run the baselines
# Train and Test CNC using Individual Token Baseline Model
# cnc_train.csv // cnc_test.csv files must exist in data/splits folder
sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_tokbase.py \
--dataset_name cnc --model_name_or_path bert-base-cased \
--output_dir outs/cnc/dev --label_column_name ce_tags \
--num_train_epochs 20 --per_device_train_batch_size 4 \
--per_device_eval_batch_size 32 --do_train_val --do_predict --do_train

### Signal Prediction
# No baseline model implemented! Just take random, using `random_st2.py` for now