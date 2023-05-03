##### TABLE 5 #####
### CNC Pre-trained Model, flagged by --model_name_or_path outs/all
K=5
SEED=42
echo "finetune and testing by folds"

# CNC --> CTB Bal (Finetune & Apply)
INPUT_CSV='./data/CTB_forCASE_rsampled.csv'
SAVE_DIR='./data/ctb_r_folds'
for i in `seq 1 $K`
do
    echo "$i"
    sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_st1.py \
    --task_name cola --train_file $SAVE_DIR/train_fold$i.csv --do_train \
    --validation_file $SAVE_DIR/test_fold$i.csv --do_eval \
    --model_name_or_path outs/all \
    --output_dir outs/ctb_r_ptm/folds/fold$i --overwrite_output_dir \
    --num_train_epochs 2 --save_steps 50000 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 32
done

# CNC --> PDTB (Finetune & Apply)
INPUT_CSV='./data/pdtb_mixed_resolved_forCASE_final.csv'
SAVE_DIR='./data/pdtb_folds'
for i in `seq 1 $K`
do
    echo "$i"
    sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_st1.py \
    --task_name cola --train_file $SAVE_DIR/train_fold$i.csv --do_train \
    --validation_file $SAVE_DIR/test_fold$i.csv --do_eval \
    --model_name_or_path outs/all \
    --output_dir outs/pdtb_ptm/folds/fold$i --overwrite_output_dir \
    --num_train_epochs 2 --save_steps 50000 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 32
done