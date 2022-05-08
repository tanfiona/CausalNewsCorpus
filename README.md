# CausalNewsCorpus (CNC)
This repository contains the model and data files for our corpus and paper titled ["The Causal News Corpus: Annotating Causal Relations in Event Sentences from News"](http://arxiv.org/abs/2204.11714). 

We invite you to participate in the CASE-2022 Shared Task: Event Causality Identification with Causal News Corpus. The task is being held as part of the [5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2022)](https://emw.ku.edu.tr/case-2022/). All participating teams will be able to publish their system description paper in the workshop proceedings published by ACL.

Make your submissions at our [Codalab Page](https://codalab.lisn.upsaclay.fr/competitions/2299).

<b>Subtask 1: Causal Event Classification</b> -- Does an event sentence contain any cause-effect meaning?<br>
<b>Subtask 2: Cause-Effect-Signal Span Detection</b> -- Which consecutive spans correspond to cause, effect or signal per causal sentence?

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

### Running BERT baseline:
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

This baseline corresponds to [Codalab submission](https://codalab.lisn.upsaclay.fr/competitions/2299#results) by "tanfiona". For LSTM baseline, submissions are by "hansih".


| # | User     | Date of Last Entry | Recall     | Precision  | F1         | Accuracy   | MCC        |
|:-:|----------|--------------------|------------|------------|------------|------------|------------|
| 1 | tanfiona | 03/23/22           | 0.8652 (1) | 0.8063 (1) | 0.8347 (1) | 0.8111 (1) | 0.6172 (1) |
| 2 | hansih   | 03/13/22           | 0.7303 (2) | 0.7514 (2) | 0.7407 (2) | 0.7183 (2) | 0.4326 (2) |

<br>

### Expected Output:
The model and parameters will be saved in the specified `--output_dir`. Alongwhich, `all_results.json` will reflect the metrics of the run. The Huggingface trainer will also automatically generate a model and results summary `README.md` file in the specified output folder.

<br>

# Subtask 2
### Data:
Within the `data` folder, we provide the datasets:
* `train_subtask2.csv`: Train set (n=133) with gold labels.
* `dev_subtask2_text.csv`: Development set (n=14) without gold labels.

The following datasets will be released as we progress along the shared task timeline:
* `dev_subtask2.csv`: Development set (n=14) with gold labels.
* `test_subtask2_text.csv`: Test set (n=TBC) without gold labels.

We are in the midst of annotating more examples. During final testing phase, we will release more train and dev examples which can be used to train your model. The data format will be exactly the same, so you can first design your models to train and test on currently available train and dev sets.
<br>

### Running BERT baseline:
To perform Cause-Effect Span Detection, we used a token classification baseline model adapted from [Huggingface's `run_ner_no_trainer.py` script](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner_no_trainer.py). 

The adapted script is part of the Causal Text Mining Benchmark titled [UniCausal](https://github.com/tanfiona/UniCausal). Due to anonymity requirements, UniCausal is not yet publicly released. However, you may contact Fiona (tan.f[at]u.nus.edu) for a private copy. See `run_st2.sh` for experiments.

We did not implement a Signal Span Detection baseline yet. Instead, we used the random generator within `random_st2.py` to obtain Signal predictions.

This baseline corresponds to [Codalab submission](https://codalab.lisn.upsaclay.fr/competitions/2299#results) by "tanfiona".

| # | User     | Date of Last Entry | Recall  | Precision  | F1  | Accuracy  | MCC  |
|:-:|----------|--------------------|---------|------------|-----|-----------|------|
| 1 | tanfiona | x                  | x       | x          | x   | x         | x    |

### Expected Output:
The model and parameters will be saved in the specified `--output_dir`. Alongwhich, a log file will reflect the metrics of the run. If `--do_predict` was opted, `span_predictions.txt` file for the predictions will also appear in this folder. The predictions are in BIO format, so you need to convert them back to a tagged sentence format (`text_w_pairs` column). Use helper functions from `src/format_st2.py` to help you.

# Cite Us
Our paper (on Subtask 1) has been accepted to LREC 2022. We hope to see you there!

If you used this repository or our corpus, please do cite us as follows:
```
@inproceedings{tan-etal-2022-causal,
    title = "The Causal News Corpus: Annotating Causal Relations in Event Sentences from News",
    author = "Tan, Fiona Anting and Hürriyetoğlu, Ali and Caselli, Tommaso and Oostdijk, Nelleke and Nomoto, Tadashi and Hettiarachchi, Hansi and Ameer, Iqra and Uca, Onur and Liza, Farhana Ferdousi and Hu, Tiancheng",
    booktitle = "Proceedings of the 13th Language Resources and Evaluation Conference",
    month = June,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    language = "English"
}
```

<br>

### Contact Us
* Fiona Anting Tan, tan.f[at]u.nus.edu
* Hansi Hettiarachchi, hansi.hettiarachchi[at]mail.bcu.ac.uk
