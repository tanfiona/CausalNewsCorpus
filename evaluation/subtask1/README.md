# CASE-2022 Event Causality Subtask 1 - Evaluation Script

## Submission instructions
The script takes one prediction file as the input. Your submission file must be a JSON file which is then zipped. We will only take the first file in the zip folder, so do not zip multiple files together.

```
{"index": 0, "prediction": 1}
{"index": 1, "prediction": 0}
```

A sample file is available [here](sample/input/res/submission.json).

❗Note1❗ Please ensure that the index order in the submission file is the same as the order in the reference data, available [here](sample/input/ref/truth.csv). 

## Testing the Script Offline
The evaluation script can run offline using the following command.
```
python evaluate.py $input $output
```

The path to the input directory should be provided as the $input argument, and the path to the output directory should be 
provided as $output. 

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
