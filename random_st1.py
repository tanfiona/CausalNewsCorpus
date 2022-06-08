# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:56:50 2022

@author: Superhhu
"""
import numpy as np
import pandas as pd
import random
import json
from random import choices
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
random.seed(42)


def get_random_predictions(train_file, test_file, iterations=1000):
    # Obtain distribution from train set
    df=pd.read_csv(train_file)
    total_label=df['label'].to_numpy()
    total_ones=np.sum(total_label)
    total_zeros=len(total_label)-total_ones

    # Apply onto test set
    df=pd.read_csv(test_file)
    random_label=choices([0,1], [total_zeros,total_ones],k=len(df))

    # Save
    preds = []
    for i,pred in enumerate(random_label):
        preds.append({'index':i,'prediction':int(pred)})
    save_file_path = 'outs/submission_random_st1.json'
    with open(save_file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in preds))


def evaluate_random_predictions(train_file, test_file, iterations=1000):
    # Obtain distribution from train set
    df=pd.read_csv(train_file)
    total_label=df['label'].to_numpy()
    total_ones=np.sum(total_label)
    total_zeros=len(total_label)-total_ones

    # Apply onto test set
    df=pd.read_csv(test_file)
    total_label=df['label'].to_numpy()

    totalprecision=[]
    totalrecall=[]
    totalfscore=[]
    totalmatthews=[]
    totalaccuracy=[]

    for i in range(iterations):
        random_label=choices([0,1], [total_zeros,total_ones],k=len(total_label))
        precision,recall,fscore,_=precision_recall_fscore_support(total_label,random_label,average='binary')
        matthews=matthews_corrcoef(total_label,random_label)
        accuracy=accuracy_score(total_label,random_label)
        totalprecision.append(precision)
        totalrecall.append(recall)
        totalfscore.append(fscore)
        totalmatthews.append(matthews)
        totalaccuracy.append(accuracy)

    print(np.mean(totalprecision))
    print(np.mean(totalrecall))
    print(np.mean(totalfscore))
    print(np.mean(totalmatthews))
    print(np.mean(totalaccuracy))


if __name__ == "__main__":
    train_file = "data/train_subtask1.csv"
    test_file = "data/test_subtask1.csv"
    get_random_predictions(train_file, test_file)
    evaluate_random_predictions(train_file, test_file)