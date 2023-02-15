## Dataset Description

#### Subtask 1 

Columns:
* index [str] : example unique id
* text [str] : example input text
* label [int] : target causal label (1 for Causal, 0 for Not Causal)
* agreement [float] : proportion of annotators supporting the vote
* num_votes [int] : number of expert labels considered 
* sample_set [str] : subset name


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

#### Evaluation Phase (Apr 15, 2022 -- Aug 01, 2022)

###### Subtask 1
* Train: train_subtask1.csv
* Test: dev_subtask1_text.csv

###### Subtask 2
* Train: train_subtask2.csv
* Test: dev_subtask2_text.csv

#### Testing Phase (Aug 01, 2022 -- Aug 31, 2022)

###### Subtask 1
* Train: train_subtask1.csv & dev_subtask1.csv
* Test: test_subtask1_text.csv

###### Subtask 2
* Train: train_subtask2.csv & dev_subtask2.csv
* Test: test_subtask2_text.csv

Note: We will not release the test labels this year as we intend to rerun this Shared Task next year.
