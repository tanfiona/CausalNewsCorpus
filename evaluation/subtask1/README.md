# CASE-2022 Event Causality Subtask 1 - Evaluation Script

## Submission instructions
The script takes one prediction file as the input. Your submission file must be a JSON file which is then zipped. We will only take the first file in the zip folder, so do not zip multiple files together.

{"index": 0, "prediction": 1}<br>
{"index": 1, "prediction": 0}<br>
...

A sample file is available [here](https://github.com/tanfiona/CausalNewsCorpus/blob/4499ae7492b55cd715eb6b1fbd4e834914e6c3cd/evaluation/sample/input/res/submission.json). Also, make sure that the index order in the submission file is the same as the 
order in the original test data. 

## Testing the Script Offline
The evaluation script can run offline using the following command.
```
python evaluate.py $input $output
```

The path to the input directory should be provided as the $input argument, and the path to the output directory should be 
provided as $output. 

The input and output directories must match the Codalab format, as shown below. The input directory must contain two subdirectories: 
'ref' with the reference ground truth dataset and 'res' with the results file. The output will be written to
 the 'scorer.txt' in the output directory. 

```
input/
 |- ref/
     |- truth.csv
 |- res/
     |- submission.json
output/
 |- scorer.txt
```