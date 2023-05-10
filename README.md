# Causal News Corpus
<img align="left" height=200 src="imgs/CNC_Logo.PNG">

This repository contains the datasets and models for the Causal News Corpus (CNC). Our dataset is published in the LREC paper titled ["The Causal News Corpus: Annotating Causal Relations in Event Sentences from News"](https://aclanthology.org/2022.lrec-1.246/) and featured in the ["Event Causality Identification with Causal News Corpus - Shared Task 3, CASE 2022"](https://aclanthology.org/2022.case-1.28/).

CNC works on two subtasks:

<b>[Subtask 1: Causal Event Classification](#subtask-1-causal-event-classification)</b> -- Does an event sentence contain any cause-effect meaning?<br>
<b>[Subtask 2: Cause-Effect-Signal Span Detection](#subtask-2-cause-effect-signal-span-detection)</b> -- Which consecutive spans correspond to cause, effect or signal per causal sentence? (Annotation in progress)

<br>

Causality is a core cognitive concept and appears in many natural language processing (NLP) works that aim to tackle inference and understanding. Generally, a causal relation is a semantic relationship between two arguments known as cause and effect, in which the occurrence of one (cause argument) causes the occurrence of the other (effect argument). The Figure below illustrates some sentences that are marked as <em>Causal</em> and <em>Non-causal</em> respectively.

| <img align="center" height=250 src="imgs/EventCausality_Subtask1_Examples3.png"> | 
|:--:| 
| *Annotated examples from Causal News Corpus. Causes are in pink, Effects in green and Signals in yellow. Note that both Cause and Effect spans must be present within one and the same sentence for us to mark it as <em>Causal</em>.* |

# Shared Task

We are hosting a shared task, known as "Event Causality Identification with Causal News Corpus". The task is part of the [6th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2023) @ RANLP](https://emw.ku.edu.tr/case-2023/). 

<b> Participate here!!!</b> (Please use "V2" data)
* [2023 Codalab Competition Page](https://codalab.lisn.upsaclay.fr/competitions/11784)

<br>

This is the second iteration of this shared task, links to last year's documentation and competition page are as follows:
* [Proceedings on ACL (@ EMNLP 2022)](https://aclanthology.org/volumes/2022.case-1/)
* [2022 Codalab Competition Page](https://codalab.lisn.upsaclay.fr/competitions/2299)
* [2022 Final Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/2299#results)


<br>

# Subtask 1: Causal Event Classification

### Data:
Datasets are available under the `data` folder. For 2023 Shared Task, please use `V2`. <b>Target column: `label`</b>

### Running BERT baseline:
Given a `<train.csv>` and `<val.csv>` file with columns `index`,`text`,`label` (`label` values should be in 0,1 int format), use our [`run_st1.py`](run_st1.py) script to train, evaluate and predict using `--do_train`, `--do_eval` and `--do_predict` flags respectively.

```
sudo python3 run_st1.py \
--task_name cola --train_file <train.csv> --do_train \
--validation_file <val.csv> --do_eval \
--test_file <val.csv> --do_predict \
--num_train_epochs 10 --save_steps 50000 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--model_name_or_path bert-base-cased \
--output_dir outs --overwrite_output_dir
```

Further experiments are also available in `run_st1.sh` for reference. Within which, we also conducted experiments with the two external corpus (CTB and PDTB V3.0). More details are described in our paper.

KFolds script is available in `kfolds.sh` which creates user-specified number of fold sets and runs the Train and Eval function over each fold. In our paper, we set K=5.

For V2 dev: This baseline corresponds to [Codalab submission](https://codalab.lisn.upsaclay.fr/competitions/11784#results) by "tanfiona".

| # | User     | Date of Last Entry | Recall     | Precision  | F1         | Accuracy   | MCC        |
|:-:|----------|--------------------|------------|------------|------------|------------|------------|
| 1 | tanfiona | 05/03/23           | 0.8378     | 0.8470 	  | 0.8424    | 0.8294    	| 0.6566     |

For V1 dev: This baseline corresponds to [Codalab submission](https://codalab.lisn.upsaclay.fr/competitions/2299#results) by "tanfiona". For LSTM baseline, submissions are by "hansih".

| # | User     | Date of Last Entry | Recall     | Precision  | F1         | Accuracy   | MCC        |
|:-:|----------|--------------------|------------|------------|------------|------------|------------|
| 1 | tanfiona | 03/23/22           | 0.8652  | 0.8063 | 0.8347  | 0.8111  | 0.6172 |
| 2 | hansih   | 03/13/22           | 0.7303  | 0.7514  | 0.7407  | 0.7183  | 0.4326 |

<br>

# Subtask 2: Cause-Effect-Signal Span Detection
### Data:
Datasets are available under the `data` folder. For 2023 Shared Task, please use `V2`. <b>Target column: `causal_text_w_pairs`</b>

To avoid revealing causal annotations for Subtask 1, we will be receiving span predictions on ALL sentences that tallies with the `_grouped` format. However, in evaluation, we only evaluate against the available annotated CAUSAL sentences. See [Subtask 2's evaluation folder](evaluation/subtask2) for more information.

### Running 1Cademy baseline:
Given a `<train.csv>`, `<val.csv>` and `<test.csv>` files, use our [`run_st2.py`](run_st2.py) script to train+evaluate and predict using `--do_train` and `--do_test` flags respectively.

```
python run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 10 \
  --num_warmup_steps 200 \
  --output_dir "outs/baseline" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 8 \
  --report_to wandb \
  --task_name ner \
  --do_train --do_test \
  --train_file <train.csv> \
  --validation_file <val.csv> \
  --test_file <test.csv> \
  --weight_decay 0.005 \
  --use_best_model
```
Further experiments are also available in `run_st2.sh` for reference. Within which, we also generate augments, provided in the data folder.
This baseline corresponds to [Codalab submission](https://codalab.lisn.upsaclay.fr/competitions/2299#results) by "tanfiona".

For V2 dev: This baseline corresponds to [Codalab submission](https://codalab.lisn.upsaclay.fr/competitions/11784#results) by "tanfiona".

| # | User     | Date of Last Entry | Recall     | Precision  | F1         |
|:-:|----------|--------------------|------------|------------|------------|
| 1 | tanfiona | 05/01/23           | 0.6632     | 0.5948 	  | 0.6271     |

For an alternate starting model, consider adapting the token classification baseline model from [Huggingface's `run_ner_no_trainer.py` script](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner_no_trainer.py). 

<br>

# Cite Us
If you used this repository or our corpus, please do cite us as follows:

##### On Subtask 1:
```
@inproceedings{tan-etal-2022-causal,
    title = "The Causal News Corpus: Annotating Causal Relations in Event Sentences from News",
    author = {Tan, Fiona Anting  and
      H{\"u}rriyeto{\u{g}}lu, Ali  and
      Caselli, Tommaso  and
      Oostdijk, Nelleke  and
      Nomoto, Tadashi  and
      Hettiarachchi, Hansi  and
      Ameer, Iqra  and
      Uca, Onur  and
      Liza, Farhana Ferdousi  and
      Hu, Tiancheng},
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.246",
    pages = "2298--2310",
    abstract = "Despite the importance of understanding causality, corpora addressing causal relations are limited. There is a discrepancy between existing annotation guidelines of event causality and conventional causality corpora that focus more on linguistics. Many guidelines restrict themselves to include only explicit relations or clause-based arguments. Therefore, we propose an annotation schema for event causality that addresses these concerns. We annotated 3,559 event sentences from protest event news with labels on whether it contains causal relations or not. Our corpus is known as the Causal News Corpus (CNC). A neural network built upon a state-of-the-art pre-trained language model performed well with 81.20{\%} F1 score on test set, and 83.46{\%} in 5-folds cross-validation. CNC is transferable across two external corpora: CausalTimeBank (CTB) and Penn Discourse Treebank (PDTB). Leveraging each of these external datasets for training, we achieved up to approximately 64{\%} F1 on the CNC test set without additional fine-tuning. CNC also served as an effective training and pre-training dataset for the two external corpora. Lastly, we demonstrate the difficulty of our task to the layman in a crowd-sourced annotation exercise. Our annotated corpus is publicly available, providing a valuable resource for causal text mining researchers.",
}
```

##### On Both Subtasks:
```
@inproceedings{tan-etal-2022-event,
    title = "Event Causality Identification with Causal News Corpus - Shared Task 3, {CASE} 2022",
    author = {Tan, Fiona Anting  and
      Hettiarachchi, Hansi  and
      H{\"u}rriyeto{\u{g}}lu, Ali  and
      Caselli, Tommaso  and
      Uca, Onur  and
      Liza, Farhana Ferdousi  and
      Oostdijk, Nelleke},
    booktitle = "Proceedings of the 5th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.case-1.28",
    pages = "195--208",
    abstract = "The Event Causality Identification Shared Task of CASE 2022 involved two subtasks working on the Causal News Corpus. Subtask 1 required participants to predict if a sentence contains a causal relation or not. This is a supervised binary classification task. Subtask 2 required participants to identify the Cause, Effect and Signal spans per causal sentence. This could be seen as a supervised sequence labeling task. For both subtasks, participants uploaded their predictions for a held-out test set, and ranking was done based on binary F1 and macro F1 scores for Subtask 1 and 2, respectively. This paper summarizes the work of the 17 teams that submitted their results to our competition and 12 system description papers that were received. The best F1 scores achieved for Subtask 1 and 2 were 86.19{\%} and 54.15{\%}, respectively. All the top-performing approaches involved pre-trained language models fine-tuned to the targeted task. We further discuss these approaches and analyze errors across participants{'} systems in this paper.",
}
```

<br>

### Contact Us
* Fiona Anting Tan, tan.f[at]u.nus.edu
* Hansi Hettiarachchi, hansi.hettiarachchi[at]mail.bcu.ac.uk
