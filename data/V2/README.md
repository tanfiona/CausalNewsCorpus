## Dataset Description

#### Subtask 1 
* `train_subtask1.csv`: Train set (n=3075) with gold labels.
* `dev_subtask1.csv`: Development set (n=340) with gold labels.
* `dev_subtask1_text.csv`: Development set (n=340) without gold labels.
* `test_subtask1_text.csv`: Test set (n=352) without gold labels.

<b>Target column: `label`</b>

#### Subtask 2

* `train_subtask2.csv`: Train set (1624 sentences, 2257 relations) with gold labels.
* `train_subtask2_grouped.csv`: Train set (n=3075) with partial gold labels.
* `dev_subtask2.csv`: Development set (185 sentences, 249 relations) with gold labels.
* `dev_subtask2_grouped.csv`: Development set (n=340) with partial gold labels.
* `dev_subtask2_text.csv`: Development set (n=340) without gold labels.
* `test_subtask2_text.csv`: Test set (n=352) without gold labels.
* `augmented_subtask2_X_train.csv`: Augmented data based on train set using `src/data_aug_st2.py` script.

To avoid revealing causal annotations for Subtask 1, we will be receiving span predictions on ALL sentences that tallies with the `_grouped` format. However, in evaluation, we only evaluate against the available annotated CAUSAL sentences. See [Subtask 2's evaluation folder](evaluation/subtask2) for more information.

<b>Target column: `causal_text_w_pairs`</b>

During final testing phase, both train and dev examples can be used to train your model. The data format will be exactly the same, so you can first design your models to train and test on currently available train and dev sets.

## Column Description

#### Subtask 2
Columns:
* corpus [str] : corpus name
* doc_id [str] : document name
* sent_id [int] : sentence id
* eg_id [int] : each sentence can have multiple relations/examples, this indicates the example id count
* index [str] : example unique id 
* text [str] : example input text 
* text_w_pairs [str] : target marked text that includes (`<ARG0>,<ARG1>,<SIGX>`) annotations
* seq_label [int] : target causal label (1 for Causal, 0 for Not Causal)
* pair_label [int] : target causal label (1 for Causal, 0 for Not Causal)
* context [str] : not relevant for CNC that works on single-sentences, to be used for non-consecutive sentence pairs
* num_sents [int] : number of sentences in text column

For Subtask 2, we recommend participants focus on the datasets with the extension `_grouped` in the file names instead. Since we are working with examples that have multiple causal relations per input text, we had to group the data such that unique texts have a single row instead of separate indexes for each relation. Namely, we group the data by "corpus, doc_id, sent_id" and keep the first "eg_id" (=0) as the main row. We then create two additional columns to reflect multiple causal relations, if available:
* causal_text_w_pairs [list] : list of up to three causal target marked text that includes (`<ARG0>,<ARG1>,<SIGX>`) annotations. if no causal relation exists, an empty list is returned.
* num_rs [int] : length of list in causal_text_w_pairs

## Shared Task
The following files are relevant for the on-going shared task on Codalab.

#### Evaluation Phase

###### Subtask 1
* Train: train_subtask1.csv
* Test: dev_subtask1_text.csv

###### Subtask 2
* Train: train_subtask2.csv
* Test: dev_subtask2_text.csv

#### Testing Phase (15 - 30 Jun 2023)

###### Subtask 1
* Train: train_subtask1.csv & dev_subtask1.csv
* Test: test_subtask1_text.csv

###### Subtask 2
* Train: train_subtask2.csv & dev_subtask2.csv
* Test: test_subtask2_text.csv

Note: We will not release the test labels this year as we might intend to rerun this Shared Task next year.
