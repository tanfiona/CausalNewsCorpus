## Dataset Description

### Subtask 1 

Columns:
* index [str] : example unique id
* text [str] : example input text
* label [int] : target causal label (1 for Causal, 0 for Not Causal)
* agreement [float]
* num_votes [int] : number of expert labels considered 
* sample_set [str] : subset name


### Subtask 2
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