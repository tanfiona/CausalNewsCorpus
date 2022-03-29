### We used the UniCausal (https://github.com/tanfiona/UniCausal) repository to run the baselines
### Cause-Effect Span Detection

# Train and Test CNC using Individual Token Baseline Model
sudo CUDA_VISIBLE_DEVICES=5 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_tokbase.py \
--dataset_name cnc --model_name_or_path bert-base-cased \
--output_dir outs/dev --label_column_name ce_tags \
--num_train_epochs 20 --per_device_train_batch_size 4 \
--per_device_eval_batch_size 32 --do_train_val --do_eval --do_predict --do_train

# Apply Trained Model on CNC (without training), two equivalent ways:
# Call by using --span_val_file
sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
--span_val_file /home/fiona/CausalTM/src/data/grouped/splits/cnc_test.csv \
--model_name_or_path /home/fiona/CausalTM/src/outs/combined \
--output_dir /home/fiona/UniCausal/outs/combined/dev --alpha 1 \
--label_column_name ce_tags --per_device_eval_batch_size 32 --do_eval --do_predict
# Call by preprocessed, predefined --dataset_name
sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
--dataset_name cnc --do_train_val \
--model_name_or_path /home/fiona/CausalTM/src/outs/combined \
--output_dir /home/fiona/UniCausal/outs/combined/dev --alpha 1 \
--label_column_name ce_tags --per_device_eval_batch_size 32 --do_eval --do_predict

# Update Trained Model on CNC
sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
--dataset_name cnc --do_train_val --do_train --alpha 1 \
--num_train_epochs 20 --per_device_train_batch_size 4 \
--model_name_or_path /home/fiona/CausalTM/src/outs/combined \
--output_dir /home/fiona/UniCausal/outs/combined/dev_updated \
--label_column_name ce_tags --per_device_eval_batch_size 32 --do_eval --do_predict

### Signal Prediction
