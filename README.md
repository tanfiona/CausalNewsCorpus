### Introduction
This repository contains the model and data files for our corpus and paper titled "The Causal News Corpus".

<br>

### Data:
`all.csv` within the `data` folder reflects the 1298 examples annotated by human experts in our corpus.

<br>

### Running the code:
Given a `train.csv` and `val.csv` file with columns `index`,`text`,`label` (`label` are in 1,0 format), use `run_case.py` script to train, evaluate and predict.

##### Train and Eval:
`
sudo CUDA_VISIBLE_DEVICES=1 python3 run_case.py --task_name cola --train_file data/train.csv --validation_file data/val.csv --model_name_or_path bert-base-cased --output_dir outs --do_train --overwrite_output_dir --do_eval --num_train_epochs 5
`
##### Eval only:
`sudo CUDA_VISIBLE_DEVICES=1 python3 run_case.py --task_name cola --train_file data/train.csv --validation_file data/val2.csv --model_name_or_path outs/ --output_dir outs --overwrite_output_dir --do_eval
`

KFolds script is available in `kfolds.sh` which creates user-specified number of fold sets and runs the Train and Eval function over each fold.

<br>

### Expected Output:

The model and parameters will be saved in the specified `--output_dir`. Alongwhich, `all_results.json` will reflect the metrics of the run.

<br>

### Experiments with External Corpus

We also conducted experiments with external corpus, the commands to run the experiments are as follows:

##### Experiments with PDTB-3
Please note that PDTB-3 is a paid corpus and therefore, we are unable to release it publicly.

`
sudo CUDA_VISIBLE_DEVICES=1 python3 run_case.py --task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final.csv --validation_file data/all.csv --model_name_or_path bert-base-cased --output_dir outs/pdtb --do_train --do_eval --overwrite_output_dir --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --num_train_epochs 2
`

`
sudo CUDA_VISIBLE_DEVICES=1 python3 run_case.py --task_name cola --train_file data/pdtb_mixed_resolved_forCASE_final_rsampled.csv --validation_file data/all.csv --model_name_or_path bert-base-cased --output_dir outs/pdtb_r --do_train --do_eval --overwrite_output_dir --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --num_train_epochs 2
`

##### Experiments with CausalTimeBank
The original data source is [publicly available](https://hlt-nlp.fbk.eu/technologies/causal-timebank).

`
sudo CUDA_VISIBLE_DEVICES=1 python3 run_case.py --task_name cola --train_file data/CTB_forCASE.csv --validation_file data/all.csv --model_name_or_path bert-base-cased --output_dir outs/ctb --do_train --do_eval --overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 5
`

`
sudo CUDA_VISIBLE_DEVICES=1 python3 run_case.py --task_name cola --train_file data/CTB_forCASE_rsampled.csv --validation_file data/all.csv --model_name_or_path bert-base-cased --output_dir outs/ctb_r --do_train --do_eval --overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 5
`

<br>

### Cite Us
If you used this repository or our corpus, please do cite us as follows:
```
bib
```

##### Contact Us
Fiona Tan, tan.f[at]u.nus.edu
