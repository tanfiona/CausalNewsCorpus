import re
import sys
import json
import pandas as pd
import numpy as np
np.random.seed(42)
from src.format_st2 import get_BIO_all, get_text_w_pairs


def random_choice_incr_probability(start,end,verbose=False):
    steps = end-start
    if steps==0:
        return start
    elif steps<0:
        raise ValueError
    else:
        list_of_steps = [i+1 for i in range(steps)]
        units = sum(list(list_of_steps))
        per_unit_p = 1/units
        step_up_ps = [i*per_unit_p for i in list_of_steps]
        assert(round(sum(step_up_ps))==1)
        if verbose:
            print(step_up_ps)
        return np.random.choice(range(start,end), 1, p=step_up_ps)[0]


def get_random_ce_pred(ce_ref, verbose=False):
    pred = ['O']*len(ce_ref)
    # randomly define start points of spans
    c_start, e_start = np.random.choice(range(len(ce_ref)), 2, replace=False)

    if c_start<e_start:
        c_end = random_choice_incr_probability(c_start+1,e_start)
        e_end = random_choice_incr_probability(e_start+1,len(ce_ref))
        pred = pred[:c_start]+['C']*(c_end-c_start)+pred[c_end:e_start]+['E']*(e_end-e_start)+pred[e_end:]
    else:
        c_end = random_choice_incr_probability(c_start+1,len(ce_ref))
        e_end = random_choice_incr_probability(e_start+1,c_start)
        pred = pred[:e_start]+['E']*(e_end-e_start)+pred[e_end:c_start]+['C']*(c_end-c_start)+pred[c_end:]
    
    if verbose:
        print('Cause_loc:',(c_start, c_end),'; Effect_loc:', (e_start, e_end))
    
    return pred


def get_random_sig_pred(sig_ref, verbose=False):
    pred = ['O']*len(sig_ref)
    
    # signals can be non-consecutive
    # the probability of getting a signal is very low though, e.g. out of 10 words, usually 1
    
    for i in range(len(sig_ref)):
        is_sig = np.random.choice([True,False], 1, p=[0.1,0.9])[0]
        if is_sig:
            pred = pred[:i]+['S']+pred[i+1:]
    
    return pred


def get_random_predictions(reference_file):

    # open file
    ref_df = pd.read_csv(reference_file)
    refs = [get_BIO_all(i) for i in ref_df['text']]

    # generate random predictions
    preds = []
    for i, ref in enumerate(refs):
        tokens, ce_ref, sig_ref = ref
        ce_pred = get_random_ce_pred(ce_ref, verbose=False)
        sig_pred = get_random_sig_pred(sig_ref, verbose=False)
        preds.append({'index':i,'prediction':get_text_w_pairs(tokens, ce_pred, sig_pred)})

    # save file
    save_file_path = 'outs/submission_random_st2.json'
    with open(save_file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in preds))


if __name__ == "__main__":
    reference_file = 'data/dev_subtask2_text.csv'
    get_random_predictions(reference_file)