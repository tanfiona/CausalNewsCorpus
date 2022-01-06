# PDTB --> CNC
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv --validation_file data/all.csv \
--model_name_or_path bert-base-cased --output_dir outs/pdtb --do_train --do_eval \
--overwrite_output_dir --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --num_train_epochs 2

# CTB --> CNC
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/CTB_forCASE.csv --validation_file data/all.csv \
--model_name_or_path bert-base-cased --output_dir outs/ctb --do_train --do_eval \
--overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 5

# PDTB --> CTB (Apply only)
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv \
--validation_file data/CTB_forCASE.csv \
--model_name_or_path outs/pdtb --output_dir outs/pdtb/ctb --do_eval \
--overwrite_output_dir --per_device_eval_batch_size 32

# PDTB --> CTB Bal (Apply only)
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv \
--validation_file data/CTB_forCASE_rsampled.csv \
--model_name_or_path outs/pdtb --output_dir outs/pdtb/ctb_r --do_eval \
--overwrite_output_dir --per_device_eval_batch_size 32

# CTB Bal --> PDTB (Apply only)
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/CTB_forCASE_rsampled.csv \
--validation_file data/pdtb_mixed_resolved_forCASE_final.csv \
--model_name_or_path outs/ctb_r --output_dir outs/ctb_r/pdtb --do_eval \
--overwrite_output_dir --per_device_eval_batch_size 64

# CNC --> CTB
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/all_train.csv --validation_file data/CTB_forCASE.csv \
--model_name_or_path bert-base-cased --output_dir outs/all --do_train --do_eval \
--overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 5

# CNC --> CTB Bal (Apply only)
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/all_train.csv \
--validation_file data/CTB_forCASE_rsampled.csv \
--model_name_or_path outs/all --output_dir outs/all/ctb_r --do_eval \
--overwrite_output_dir --per_device_eval_batch_size 32

# CNC --> PDTB (Apply only)
sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
--task_name cola --train_file data/all_train.csv \
--validation_file data/pdtb_mixed_resolved_forCASE_final.csv \
--model_name_or_path outs/all --output_dir outs/all/pdtb --do_eval \
--overwrite_output_dir --per_device_eval_batch_size 64

