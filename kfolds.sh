#!/bin/sh

# declare vars, no spacing
INPUT_CSV='./data/all.csv'
K=5
SEED=42
SAVE_DIR='./data/folds'

# gen kfolds
echo "generating folds"
sudo /home/fiona/anaconda3/envs/torchgeom/bin/python3 gen_kfolds.py \
--input_csv $INPUT_CSV --k $K --seed $SEED --save_dir $SAVE_DIR

# run per kfold
echo "train and testing by folds"
for i in `seq 1 $K`
do
    echo "$i"
    sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_case.py \
    --task_name cola --train_file $SAVE_DIR/train_fold$i.csv --validation_file $SAVE_DIR/test_fold$i.csv \
    --model_name_or_path bert-base-cased --output_dir outs/folds/fold$i \
    --overwrite_output_dir --do_train --do_eval --num_train_epochs 5
done