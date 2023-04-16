import re
import json
from ast import literal_eval
import evaluate
import itertools


def get_combinations(list1,list2):
    return [list(zip(each_permutation, list2)) for each_permutation in itertools.permutations(list1, len(list2))]


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


def format_results(metric):
    
    results = metric.compute()
    
    final_results = {}
    for key in ['Cause','Effect','Signal']:
        final_results[key]=results[key[0]]
    
    count_metrics = ['TP','FP','FN','BE','LE','LBE']
    final_results['Overall'] = {}
    for c in count_metrics:
        final_results['Overall'][c]=results[c]
    for k in results.keys():
        if 'overall_' in k:
            new_k = '_'.join(k.split('_')[1:])
            final_results['Overall'][new_k]=results[k]
    
    return final_results


def keep_best_combinations_only(row, refs, preds):
    best_metric = -1
    best_pair = None

    for points in get_combinations(row.id, row.id):
        
        # initialise
        metric = evaluate.load("hpi-dhc/FairEval")

        # add rounds
        for a,b in list(points):
            _, ce_ref, sig_ref = refs[a]
            _, ce_pred, sig_pred = preds[b]
            metric.add(
                prediction=ce_pred,
                reference=ce_ref 
            )
            metric.add(
                prediction=sig_pred,
                reference=sig_ref 
            )
    
        # compute
        results = metric.compute()
        key_metric = float(results['overall_f1'])
        if key_metric>best_metric:
            # overwrite if best
            best_metric=key_metric
            best_pair = points
            
    return best_pair


def combine_dicts(d1,d2):
    the_keys = ['Cause','Effect','Signal']
    metrics = ['precision','recall','f1']    
    d0 = {k:{i:0 for i in metrics} for k in the_keys}
    
    for k in the_keys:
        total_weight=0
        if k in d1.keys():
            for m in metrics:
                d0[k][m]+=d1[k][m]*d1[k]['number']
            total_weight+=d1[k]['number']
        if k in d2.keys():
            for m in metrics:
                d0[k][m]+=d2[k][m]*d2[k]['number']
            total_weight+=d2[k]['number']
            
        for m in metrics:
            d0[k][m]/=total_weight
            d0[k]['number']=total_weight
    
    d0['Overall'] = {i:0 for i in metrics}
    total_weight = 0
    for key in the_keys:
        ddict = d0[key]
        for i in metrics:
            d0['Overall'][i]+=ddict[i]*ddict['number']
        total_weight+=ddict['number']
    for k,v in d0['Overall'].items():
        d0['Overall'][k]=v/total_weight
    
    d0['Overall']['number'] = d1['Overall']['number']+d2['Overall']['number']
    d0['Overall']['accuracy'] = d1['Overall']['accuracy']*d1['Overall']['number']
    d0['Overall']['accuracy'] += d2['Overall']['accuracy']*d2['Overall']['number']
    d0['Overall']['accuracy'] /= d0['Overall']['number']
    
    return d0


def main(ref_df, pred_list, calculate_best_combi=True):
    # Convert
    ref_df, pred_list = keep_relevant_rows_and_unstack(ref_df, pred_list)
    assert(len(pred_list)==len(ref_df))
    refs = [get_BIO_all(i) for i in ref_df['text_w_pairs']]
    preds = [get_BIO_all(i) for i in pred_list]
    
    # Group
    if calculate_best_combi:
        grouped_df = ref_df.copy()
        grouped_df['id'] = [[i] for i in grouped_df.index]
        grouped_df = grouped_df.groupby(['corpus','doc_id','sent_id'])[['eg_id','id']].agg({'eg_id':'count','id':'sum'}).reset_index()
        grouped_df = grouped_df[grouped_df['eg_id']>1]
        req_combi_ids = [item for sublist in grouped_df['id'] for item in sublist]
    else:
        grouped_df = None
        req_combi_ids = []

    # For examples that DO NOT require combination search
    regular_ids = list(set(range(len(preds)))-set(req_combi_ids))

    metric = evaluate.load("hpi-dhc/FairEval")

    for i in regular_ids:
        _, ce_ref, sig_ref = refs[i]
        _, ce_pred, sig_pred = preds[i]
        metric.add(
            prediction=ce_pred,
            reference=ce_ref 
        )
        metric.add(
            prediction=sig_pred,
            reference=sig_ref 
        )

    multi_metric = evaluate.load("hpi-dhc/FairEval")
    if grouped_df is not None:
        # For examples that require combination search
        for _, row in grouped_df.iterrows():
            best_pair = keep_best_combinations_only(row, refs, preds)
            for a,b in list(best_pair):
                _, ce_ref, sig_ref = refs[a]
                _, ce_pred, sig_pred = preds[b]
                metric.add(
                    prediction=ce_pred,
                    reference=ce_ref 
                )
                metric.add(
                    prediction=sig_pred,
                    reference=sig_ref 
                )
                multi_metric.add(
                    prediction=ce_pred,
                    reference=ce_ref 
                )
                multi_metric.add(
                    prediction=sig_pred,
                    reference=sig_ref 
                )

    result = format_results(metric)
    multi_result = format_results(multi_metric)

    return result, multi_result
    

def keep_relevant_rows_and_unstack(ref_df, predictions):
    
    # Keep only causal examples
    predictions_w_true_labels = []
    eg_id_counter = []
    for i, row in ref_df.iterrows():
        if row.num_rs>0:
            p = predictions[i]
            if len(p)>row.num_rs:
                # Note if you predict more than the number of relations we have, we only keep the first few.
                # We completely ignore the subsequent predictions.
                p = p[:row.num_rs]
            elif len(p)<row.num_rs:
                # Incorporate dummy predictions if there are insufficient predictions
                p.extend([row.text]*(row.num_rs-len(p)))
            predictions_w_true_labels.extend(p)
            eg_id_counter.extend(list(range(row.num_rs)))
    ref_df = ref_df[ref_df['num_rs']>0].reset_index(drop=True)
    
    # Expand into single rows
    ref_df['causal_text_w_pairs'] = ref_df['causal_text_w_pairs'].apply(lambda x: literal_eval(x))
    ref_df = ref_df.explode('causal_text_w_pairs')
    ref_df = ref_df.rename(columns={'causal_text_w_pairs':'text_w_pairs'})
    ref_df['eg_id'] = eg_id_counter
    
    return ref_df.reset_index(drop=True), predictions_w_true_labels