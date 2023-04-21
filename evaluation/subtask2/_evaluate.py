#!/usr/bin/env python
import os
import sys
import pandas as pd
from utils_eval_st2 import main, read_predictions


"""
Test locally using:

    D:\61 Challenges\2022_CASE_\CausalNewsCorpus\CausalNewsCorpus>
    python evaluation/subtask2/_evaluate.py \
        evaluation/subtask2/sample/input \
            evaluation/subtask2/sample/output

"""

# Codalab Run
input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    raise FileNotFoundError("{0} doesn't exist!".format(submit_dir))

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    # read ground truth
    truth_file_name = os.listdir(truth_dir)[0] #"truth.csv"
    truth_file = os.path.join(truth_dir, truth_file_name)
    truth = pd.read_csv(truth_file, sep=",", encoding="utf-8")

    # read predictions
    submit_file_name = os.listdir(submit_dir)[0] #"submission.json"
    submission_answer_file = os.path.join(submit_dir, submit_file_name)
    predictions = read_predictions(submission_answer_file)

    if len(truth) != len(predictions):
        raise IndexError("Number of entries in the submission.json do not match with the ground truth entry count!")
    else:
        # evaluate
        result, multi_result = main(truth, predictions, calculate_best_combi=True)
        # write to output file
        level_order = ['Overall','Cause','Effect','Signal']
        columns_order = ['Recall', 'Precision', 'F1']

        # output_file.write("### All Examples\n")
        for level in level_order:
            ddict = result[level]
            for k in columns_order:
                if k.lower() in ddict:
                    v = ddict[k.lower()]
                    if level=='Overall':
                        output_file.write("{0}:{1}\n".format(k.title(),v))
                    else:
                        output_file.write("{0}_{1}:{2}\n".format(level,k.title(),v))
                else:
                    continue
        
        # output_file.write("\n### Examples with Multiple Relations\n")
        for level in level_order:
            ddict = multi_result[level]
            for k in columns_order:
                if k.lower() in ddict:
                    v = ddict[k.lower()]
                    if level=='Overall':
                        output_file.write("MultiEgs_{0}:{1}\n".format(k.title(),v))
                    else:
                        output_file.write("MultiEgs_{0}_{1}:{2}\n".format(level,k.title(),v))
                else:
                    continue

    output_file.close()


