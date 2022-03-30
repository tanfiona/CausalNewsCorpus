#!/usr/bin/env python
import json
import sys
import os
import re
import os.path
import pandas as pd
from datasets import load_metric


def read_predictions(submission_file):
    predictions = []
    with open(submission_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                predictions.append(json.loads(line)['prediction'])
    return predictions


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub('</*[A-Z]+\d*>','',tok) 

def get_BIO(text_w_pairs):
    tokens = []
    ce_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):

        # Replace if special
        if '<ARG0>' in tok:
            tok = re.sub('<ARG0>','',tok)
            tag = 'B-C'
            next_tag = 'I-C'
        elif '</ARG0>' in tok:
            tok = re.sub('</ARG0>','',tok)
            tag = 'I-C'
            next_tag = 'O'
        elif '<ARG1>' in tok:
            tok = re.sub('<ARG1>','',tok)
            tag = 'B-E'
            next_tag = 'I-E'
        elif '</ARG1>' in tok:
            tok = re.sub('</ARG1>','',tok)
            tag = 'I-E'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        ce_tags.append(tag)
        tag = next_tag
    
    return tokens, ce_tags

### S
def get_BIO_sig(text_w_pairs):
    tokens = []
    s_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):
        # Replace if special
        if '<SIG' in tok:
            tok = re.sub('<SIG([A-Z]|\d)*>','',tok)
            tag = 'B-S'
            next_tag = 'I-S'
            if '</SIG' in tok: # one word only
                tok = re.sub('</SIG([A-Z]|\d)*>','',tok)
                next_tag = 'O'

        elif '</SIG' in tok:
            tok = re.sub('</SIG([A-Z]|\d)*>','',tok)
            tag = 'I-S'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        s_tags.append(tag)
        tag = next_tag
    
    return tokens, s_tags


def get_BIO_all(text_w_pairs):
    tokens, ce_tags = get_BIO(text_w_pairs)
    tokens_s, s_tags = get_BIO_sig(text_w_pairs)
    assert(tokens==tokens_s)
    assert(len(ce_tags)==len(s_tags)==len(tokens))
    return tokens, ce_tags, s_tags


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
    actuals = truth['text_w_pairs'].tolist()

    # read predictions
    submit_file_name = os.listdir(submit_dir)[0] #"submission.json"
    submission_answer_file = os.path.join(submit_dir, submit_file_name)
    predictions = read_predictions(submission_answer_file)

    if len(actuals) != len(predictions):
        raise IndexError("Number of entries in the submission.json do not match with the ground truth entry count!")
    # evaluate
    else:
        # Initialise metric
        ce_metric = load_metric('seqeval')
        sig_metric = load_metric('seqeval')

        # Add values
        for i, pred in enumerate(predictions):
            _, ce_ref, sig_ref = get_BIO_all(actuals[i])
            _, ce_pred, sig_pred = get_BIO_all(pred)
            ce_metric.add(
                prediction=ce_pred,
                reference=ce_ref 
            )
            sig_metric.add(
                prediction=sig_pred,
                reference=sig_ref 
            )

        # Compute
        final_results = {}
        metrics = ['precision','recall','f1','accuracy']
        final_results['Overall'] = {i:0 for i in metrics}

        results = sig_metric.compute()
        final_results['Signal'] = results['S']
        final_results['Overall']['accuracy'] += results['overall_accuracy']

        results = ce_metric.compute()
        final_results['Cause'] = results['C']
        final_results['Effect'] = results['E']
        total_weight = 0
        for key in ['Cause','Effect','Signal']:
            ddict = final_results[key]
            for i in metrics[:-1]:
                final_results['Overall'][i]+=ddict[i]*ddict['number']
            total_weight+=ddict['number']
        for k,v in final_results['Overall'].items():
            final_results['Overall'][k]=v/total_weight
        final_results['Overall']['accuracy'] += results['overall_accuracy']
        final_results['Overall']['accuracy'] /= 2

        for level,ddict in final_results.items():
            for k,v in ddict.items():
                if level=='Overall':
                    output_file.write("{0}:{1}\n".format(k.title(),v))
                else:
                    output_file.write("{0}_{1}:{2}\n".format(level,k.title(),v))

    output_file.close()
