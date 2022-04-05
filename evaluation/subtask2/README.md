# CASE-2022 Event Causality Subtask 2 - Evaluation Script

## Submission instructions
The script takes one prediction file as the input. Your submission file must be a JSON file which is then zipped. We will only take the first file in the zip folder, so do not zip multiple files together.

{"index": 0, "prediction": "`<ARG0>`Dissatisfied with the package`</ARG0>` , `<ARG1>`workers staged an all-nigh sit-in`</ARG1>` ."}<br>
{"index": 1, "prediction": "`<ARG1>`Three people were killed`</ARG1>` `<ARG1>`and 69 others injured`</ARG1>` `<ARG0>`in the explosion`</ARG0>` ."}<br>
...

A sample file is available [here](https://github.com/tanfiona/CausalNewsCorpus/blob/master/evaluation/subtask2/sample/input/res/submission.json). Also, make sure that the index order in the submission file is the same as the order in the original test data. The only exception is for examples with multiple relations (E.g. Multiple 'eg_id' per unique 'corpus'x'doc_id'x'sent_id'). Our code will automatically extract the combination that results in the best F1 score, so you do not need to worry about how to order the predictions of such multi-relation examples.

## Testing the Script Offline
The evaluation script can run offline using the following command.
```
python evaluate.py $input $output
```

The path to the input directory should be provided as the $input argument, and the path to the output directory should be provided as $output.

The input and output directories must match the Codalab format, as shown below. The input directory must contain two subdirectories:
'ref' with the reference ground truth dataset and 'res' with the results file. 
The output will be written to the 'scores.txt' in the output directory. 

```
input/
 |- ref/
     |- truth.csv
 |- res/
     |- submission.json
output/
 |- scores.txt
```
