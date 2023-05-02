GPU_ID=0
TASK=baseline

if [[ "$TASK" == "baseline" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st1.py \
  --task_name cola \
  --train_file data/V2/train_subtask1.csv --do_train \
  --validation_file data/V2/dev_subtask1.csv --do_eval \
  --test_file data/V2/test_subtask1.csv --do_predict \
  --model_name_or_path bert-base-cased \
  --output_dir outs/dev --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 \
  --save_strategy epoch --evaluation_strategy epoch --logging_strategy epoch \
  --load_best_model_at_end True --metric_for_best_model eval_f1 --save_total_limit 2

elif [[ "$TASK" == "test" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st1.py \
  --task_name cola \
  --test_file data/V2/dev_subtask1.csv --do_predict \
  --model_name_or_path "path_to_model" \
  --output_dir outs/dev --overwrite_output_dir True

elif [[ "$TASK" == "other" ]]; then
  ##### TABLE 3 #####
  ### Internal
  # CNC Train+Dev --> Test
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/all_train_subtask1.csv --do_train \
  --validation_file data/test_subtask1.csv --do_eval \
  --test_file data/test_subtask1.csv --do_predict \
  --model_name_or_path bert-base-cased --output_dir outs/test --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000

  # CNC Train --> Dev
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/train_subtask1.csv --do_train \
  --validation_file data/dev_subtask1.csv --do_eval \
  --test_file data/dev_subtask1.csv --do_predict \
  --model_name_or_path bert-base-cased --output_dir outs/dev --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000

  # CNC Train --> Dev
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/train_subtask1.csv --do_train \
  --validation_file data/dev_subtask1.csv --do_eval \
  --test_file data/dev_subtask1.csv --do_predict \
  --model_name_or_path bert-base-cased --output_dir outs/dev --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000

  ### External

  # PDTB --> CNC Test
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv --do_train \
  --validation_file data/test_subtask1.csv --do_eval \
  --model_name_or_path bert-base-cased --output_dir outs/pdtb --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000

  # PDTB Bal --> CNC Test
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final_rsampled.csv --do_train \
  --validation_file data/test_subtask1.csv --do_eval \
  --model_name_or_path bert-base-cased --output_dir outs/pdtb --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000

  # CTB --> CNC Test
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/CTB_forCASE.csv --do_train \
  --validation_file data/test_subtask1.csv --do_eval \
  --model_name_or_path bert-base-cased --output_dir outs/ctb --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000

  # CTB Bal --> CNC Test
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/CTB_forCASE_rsampled.csv --do_train \
  --validation_file data/test_subtask1.csv --do_eval \
  --model_name_or_path bert-base-cased --output_dir outs/ctb_r --overwrite_output_dir \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 10 --save_steps 50000


  ##### TABLE 4 #####
  ### See KFolds scripts for Within

  # CNC Train+Dev+Test
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/all_subtask1.csv --do_train \
  --validation_file data/test_subtask1.csv \
  --model_name_or_path bert-base-cased --output_dir outs/all --overwrite_output_dir \
  --per_device_train_batch_size 32 --num_train_epochs 10 --save_steps 50000

  # CNC --> CTB Bal (Apply only)
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/all_subtask1.csv \
  --validation_file data/CTB_forCASE_rsampled.csv \
  --model_name_or_path outs/all --output_dir outs/all/ctb_r --do_eval \
  --overwrite_output_dir --per_device_eval_batch_size 32

  # CNC --> PDTB (Apply only)
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/all_subtask1.csv \
  --validation_file data/pdtb_mixed_resolved_forCASE_final.csv \
  --model_name_or_path outs/all --output_dir outs/all/pdtb --do_eval \
  --overwrite_output_dir --per_device_eval_batch_size 32

  # CTB Bal --> CNC (Apply only)
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/CTB_forCASE_rsampled.csv \
  --validation_file data/all_subtask1.csv \
  --model_name_or_path outs/ctb_r --output_dir outs/ctb_r/all --do_eval \
  --overwrite_output_dir --per_device_eval_batch_size 32

  # CTB Bal --> PDTB (Apply only)
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/CTB_forCASE_rsampled.csv \
  --validation_file data/pdtb_mixed_resolved_forCASE_final.csv \
  --model_name_or_path outs/ctb_r --output_dir outs/ctb_r/pdtb --do_eval \
  --overwrite_output_dir --per_device_eval_batch_size 32

  # PDTB --> CNC (Apply only)
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv \
  --validation_file data/all_subtask1.csv \
  --model_name_or_path outs/pdtb --output_dir outs/pdtb/all --do_eval \
  --overwrite_output_dir --per_device_eval_batch_size 32

  # PDTB --> CTB Bal (Apply only)
  sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
  --task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv \
  --validation_file data/CTB_forCASE_rsampled.csv \
  --model_name_or_path outs/pdtb --output_dir outs/pdtb/ctb_r --do_eval \
  --overwrite_output_dir --per_device_eval_batch_size 32
fi