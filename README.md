# Introduction
This repository contains the model and data files for our corpus and paper titled "The Causal News Corpus". 

We invite you to participate in the CASE-2022 Shared Task: Event Causality Identification with Causal News Corpus. The task is being held as part of the [5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2022)](https://emw.ku.edu.tr/case-2022/). All participating teams will be able to publish their system description paper in the workshop proceedings published by ACL.

Make your submissions at our [Codalab Page](https://codalab.lisn.upsaclay.fr/competitions/2299).

<br>

# Subtask 1

### Data:
Within the `data` folder, we provide the datasets:
* `train_subtask1.csv`: Train set (n=2925) with gold labels.
* `dev_subtask1_text.csv`: Development set (n=323) without gold labels.
* `CTB_forCASE.csv`: Processed [CausalTimeBank](https://hlt-nlp.fbk.eu/technologies/causal-timebank) dataset.
* `CTB_forCASE_rsampled.csv`: Processed CausalTimeBank dataset and sampled for balanced class labels.

The following datasets will be released as we progress along the shared task timeline:
* `dev_subtask1.csv`: Development set (n=323) with gold labels.
* `test_subtask1_text.csv`: Test set (n=311) without gold labels.

The following datasets are used in our experiments, but not released, due to copyright issues:
* `pdtb_mixed_resolved_forCASE_final`: Processed [PDTB V3.0](https://catalog.ldc.upenn.edu/LDC2019T05) dataset.

<br>

### Running the code:
Given a `<train.csv>` and `<val.csv>` file with columns `index`,`text`,`label` (`label` values should be in 0,1 int format), use our `run_case.py` script to train, evaluate and predict using `--do_train`, `--do_eval` and `--do_predict` flags respectively.

```
sudo python3 run_case.py \
--task_name cola --train_file <train.csv> --do_train \
--validation_file <val.csv> --do_eval \
--test_file <val.csv> --do_predict \
--num_train_epochs 10 --save_steps 50000 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--model_name_or_path bert-base-cased \
--output_dir outs --overwrite_output_dir
```

KFolds script is available in `kfolds.sh` which creates user-specified number of fold sets and runs the Train and Eval function over each fold. In our paper, we set K=5.

Further experiments are also available in `run_st1.sh` for reference. Within which, we also conducted experiments with the two external corpus (CTB and PDTB V3.0). More details are described in our paper.

<br>

### Expected Output:

The model and parameters will be saved in the specified `--output_dir`. Alongwhich, `all_results.json` will reflect the metrics of the run. The Huggingface trainer will also automatically generate a model and results summary `README.md` file in the specified output folder.

<br>

# Subtask 2
### Data:
Within the `data` folder, we provide the datasets:
* `train_subtask2.csv`: Train set (n=XXX) with gold labels.
* `dev_subtask2_text.csv`: Development set (n=XXX) without gold labels.

The following datasets will be released as we progress along the shared task timeline:
* `dev_subtask2.csv`: Development set (n=XXX) with gold labels.
* `test_subtask2_text.csv`: Test set (n=XXX) without gold labels.

<br>

### Running the code:
We used baseline models from [UniCausal](https://github.com/tanfiona/UniCausal) to perform Cause-Effect Span Detection. See `run_st2,sh` for experiments.


# Cite Us
If you used this repository or our corpus, please do cite us as follows:
```
To be added
```

<br>

### Contact Us
* Fiona Tan, tan.f[at]u.nus.edu
* Hansi Hettiarachchi, hansi.hettiarachchi[at]mail.bcu.ac.uk
