#!/bin/sh

# declare vars, no spacing
INPUT_CSV='./data/CTB_forCASE_rsampled.csv'
K=5
SEED=42
SAVE_DIR='./data/ctb_r_folds'

# gen kfolds
echo "generating folds"
sudo /home/fiona/anaconda3/envs/torchgeom/bin/python3 gen_kfolds.py \
--input_csv $INPUT_CSV --k $K --seed $SEED --save_dir $SAVE_DIR

# run per kfold
echo "train and testing by folds"
for i in `seq 1 $K`
do
    echo "$i"
    sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_st1.py \
    --task_name cola --train_file $SAVE_DIR/train_fold$i.csv --do_train \
    --validation_file $SAVE_DIR/test_fold$i.csv --do_eval \
    --model_name_or_path bert-base-cased --output_dir outs/ctb_r/folds/fold$i --overwrite_output_dir \
    --num_train_epochs 10 --save_steps 50000 --per_device_train_batch_size 32 --per_device_eval_batch_size 32
done
