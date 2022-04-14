#!/usr/bin/env python
import json
import sys
import os
import os.path
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, matthews_corrcoef


def read_predictions(submission_file):
    predictions = []
    with open(submission_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                predictions.append(json.loads(line)['prediction'])
    return predictions


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
    truth_file_name = os.listdir(truth_dir)[0]  # "truth.csv"
    truth_file = os.path.join(truth_dir, truth_file_name)
    truth = pd.read_csv(truth_file, sep=",", encoding="utf-8")
    actuals = truth['label'].tolist()

    # read predictions
    submit_file_name = os.listdir(submit_dir)[0]  # "submission.json"
    submission_answer_file = os.path.join(submit_dir, submit_file_name)
    predictions = read_predictions(submission_answer_file)

    if len(actuals) != len(predictions):
        raise IndexError("Number of entries in the submission.json do not match with the ground truth entry count!")
    # evaluate
    else:
        r = recall_score(actuals, predictions)
        output_file.write("Recall:{0}\n".format(r))
        p = precision_score(actuals, predictions)
        output_file.write("Precision:{0}\n".format(p))
        f1 = f1_score(actuals, predictions)
        output_file.write("F1:{0}\n".format(f1))
        accuracy = accuracy_score(actuals, predictions)
        output_file.write("Accuracy:{0}\n".format(accuracy))
        mcc = matthews_corrcoef(actuals, predictions)
        output_file.write("MCC:{0}\n".format(mcc))

        # # debugging
        # print("Recall:{0}".format(r))
        # print("Precision:{0}".format(p))
        # print("F1:{0}".format(f1))
        # print("Accuracy:{0}".format(accuracy))
        # print("MCC:{0}".format(mcc))

    output_file.close()
