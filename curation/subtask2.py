"""
Convert curated WebAnno JSON format into CSV files
"""
import json
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from pandas import ExcelWriter

import itertools
from itertools import combinations
from kAlpha import get_result
midfix = "s" 
# midfix = "test_s"


def get_combinations(list1,list2):
    return [list(zip(each_permutation, list2)) for each_permutation in itertools.permutations(list1, len(list2))]

def sum_all_scores(metric_dict, loc):
    return metric_dict['Effect'][loc]+metric_dict['Cause'][loc]+metric_dict['Signal'][loc]+metric_dict['All'][loc]

def flatten(t):
    return [item for sublist in t for item in sublist]

def del_item_in_list(thelist, locs):
    for i,l in enumerate(sorted(locs)):
        thelist.pop(l-i)
    return thelist

def keep_best_combinations_only(num_rels, to_compare, for_kalpha, exactmatch, onesidebound, tokenoverlap):
    # sents with mismatched rels
    for sentid in [i+1 for i,r in enumerate(num_rels) if r==0]:

        # to compare : (8, 0, 0), (8, 0, 1), (8, 1, 0), (8, 1, 1)
        # choose the best combi

        indexes = [i for i, (s,a,b) in enumerate(to_compare) if s==sentid]
        if len(indexes)<=1:
            continue
        else:
            ddict = {}
            for i in indexes:
                ddict[to_compare[i]]=sum_all_scores(exactmatch,i)+sum_all_scores(onesidebound,i)+sum_all_scores(tokenoverlap,i)

            thelist = range(to_compare[i][-1]+1)
            combi_list = get_combinations(thelist,thelist)
            combi_scores = []
            for combi in combi_list:
                score = 0
                for (a,b) in combi:
                    score+=ddict[(sentid,a,b)]
                combi_scores.append(score)

            retain_index = max(range(len(combi_scores)), key=combi_scores.__getitem__)
            remove_items = [(sentid,a,b) for a,b in flatten([c for i,c in enumerate(combi_list) if i!=retain_index])]

            remove_index = [i for i,c in enumerate(to_compare) if c in remove_items]
            to_compare = del_item_in_list(to_compare, remove_index)
            for_kalpha = del_item_in_list(for_kalpha, remove_index)
            for k,v in exactmatch.items():
                exactmatch[k]=del_item_in_list(v, remove_index)
            for k,v in onesidebound.items():
                onesidebound[k]=del_item_in_list(v, remove_index)
            for k,v in tokenoverlap.items():
                tokenoverlap[k]=del_item_in_list(v, remove_index)


def get_ref_df(save_folder=None):
    # Change per run: Only amend "adds_ref_path"
    base_ref_path = r"D:\61 Challenges\2022_CASE_\Presentation\20220422 Virtual Update\pdata_for_subtask2_donesofar.csv"
    adds_ref_path = [
        # round5
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_test_s01.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_test_s02.csv",
        # round6
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s09.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s10.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s11.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s12.csv",
        # round 7
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s13.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s14.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s15.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s16.csv",
        # round 8
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s17.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s18.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s19.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s20.csv",
        # round 9
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s21.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s22.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s23.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s24.csv",
        # round 10
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s25.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s26.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s27.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s28.csv",
        # round 11,12
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s29.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s30.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s31.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s32.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s33.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s34.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s35.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s36.csv",
        # round 13 (final)
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s37.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s38.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s39.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_s40.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_test_s03.csv",
        r"D:\61 Challenges\2022_CASE_\Presentation\20220512 Meeting\manually_formatted\subtask2_test_s04.csv",
    ]

    ref_df = pd.read_csv(base_ref_path)
    for i,r in enumerate(adds_ref_path):
        ref = pd.read_csv(r)
        ref['source'] = os.path.basename(r)
        ref['Index'] = ref.index
        ref['Index']+=1
        ref_df = pd.concat([ref_df, ref], axis=0)
    ref_df['source'] = ref_df['source'].apply(lambda x: os.path.splitext(x)[0])

    if save_folder is not None:
        ref_df.to_csv(os.path.join(save_folder, f"pdata_for_subtask2_donesofar.csv"), index=False, encoding='utf-8-sig')

    return ref_df

def open_json(json_file_path, data_format=list):
    if data_format==dict or data_format=='dict':
        with open(json_file_path) as json_file:
            data = json.load(json_file)
    elif data_format==list or data_format=='list':
        data = []
        for line in open(json_file_path, encoding='utf-8'):
            data.append(json.loads(line))
    elif data_format==pd.DataFrame or data_format=='pd.DataFrame':
        data = pd.read_json(json_file_path, orient="records", lines=True)
    else:
        raise NotImplementedError
    return data

def find_sents_needed(sent2locid, search_min, search_max):
    sents_needed = []
    for k,v in sent2locid.items():
        if v<search_min:
            # keep replacing as first item
            sents_needed = [k+1]
            continue
        if v>=search_max:
            # sufficient found, exit
            break
        sents_needed.append(k+1)
    return sents_needed

def remap_index(del_begin, del_end, begin, end, verbose=False):
    if begin>del_end and end>del_end: ### [del_begin---del_end]---(begin---end)
        if verbose: print('Setting 1')
        del_counts = del_end-del_begin
        begin-=del_counts
        end-=del_counts
    elif begin<del_begin and end<del_begin: ### (begin---end)---[del_begin---del_end]
        if verbose: print('Setting 2')
        pass
    elif begin>=del_begin and end>=del_end: ### [del_begin---(begin---del_end]---end)
        if verbose: print('Setting 3')
        del_counts = del_end-del_begin
        begin=del_begin
        end-=del_counts
    elif begin>=del_begin and end<=del_end: ### [del_begin---(begin---end)---del_end]
        if verbose: print('Setting 4')
        del_counts = begin-del_begin
        begin-=del_counts
        end-=del_counts
    elif begin<=del_begin and end>=del_end:   ### (begin---[del_begin---del_end]---end)
        if verbose: print('Setting 5')
        end=(end-del_end)+del_begin
    elif begin<=del_begin and end>=del_begin:   ### (begin---[del_begin---end)---del_end]
        if verbose: print('Setting 6')
        end=del_begin
    else:
        print(f'Undefined: {(del_begin,del_end)}, {(begin, end)}')
        
    return (begin, end)

def format_report(annotations, global_errors, span2sentid):
    report = []
    cols = ['Index', 'Span1', 'Span2', 'Error', 'Annotator']

    for m in global_errors['missing']:
        sentid = m['sentid']
        text = annotations[sentid]['retrieve_info']['text']
        buffer = annotations[sentid]['retrieve_info']['begin']
        span_text = text[m['begin']-buffer:m['end']-buffer]
        report.append([sentid, span_text, None, 'Span has no link.', m['annotator']])

    for m in global_errors['no_label']:
        sentid = m['sentid']
        text = annotations[sentid]['retrieve_info']['text']
        buffer = annotations[sentid]['retrieve_info']['begin']
        span_text = text[m['begin']-buffer:m['end']-buffer]
        report.append([sentid, span_text, None, 'Span has no label.', m['annotator']])
        
    for r in global_errors['backwards']:
        cause_or_sig_label = 'Cause' if 'Cause' in r.keys() else 'Signal'
        report.append([span2sentid[r['spanid']], r[cause_or_sig_label], r['Effect'], 'Possibly reversed arrow.', r['Annotator']])


    for r in global_errors['no_cause']:
        report.append([span2sentid[r['spanid']], r['Signal'], r['Effect'], 'Missing Cause in links: Signal-->Effect.', r['Annotator']])


    for r,m1,m2 in global_errors['effects_only']:

        span1_label, span2_label = m1['label'], m2['label']
        sentid = span2sentid[m1['spanid']]
        text = annotations[sentid]['retrieve_info']['text']
        buffer = annotations[sentid]['retrieve_info']['begin']
        span1_text = text[m1['begin']-buffer:m1['end']-buffer]
        span2_text = text[m2['begin']-buffer:m2['end']-buffer] # assuming same sentence

        report.append([sentid, span1_text, span2_text, f'Only Effect in links: {span1_label}-->{span2_label}', m1['annotator']])


    for r,m1,m2 in global_errors['no_effects']:

        span1_label, span2_label = m1['label'], m2['label']
        sentid = span2sentid[m1['spanid']]
        text = annotations[sentid]['retrieve_info']['text']
        buffer = annotations[sentid]['retrieve_info']['begin']
        span1_text = text[m1['begin']-buffer:m1['end']-buffer]
        span2_text = text[m2['begin']-buffer:m2['end']-buffer] # assuming same sentence

        report.append([sentid, span1_text, span2_text, f'Missing Effect in links: {span1_label}-->{span2_label}', m1['annotator']])

    return pd.DataFrame(report, columns=cols)

def get_s_e_t_list(s_head=[], e_head=[], s_tail=[], e_tail=[], s_sig=[], e_sig=[]):
    locs, tags = [], []
    if isinstance(s_head,(np.ndarray,list)):
        locs.extend(s_head)
        tags.extend(['<ARG0>']*len(s_head))
        locs.extend(e_head)
        tags.extend(['</ARG0>']*len(e_head))
    else:
        locs.append(s_head)
        tags.append('<ARG0>')
        locs.append(e_head)
        tags.append('</ARG0>')

    if isinstance(s_tail,(np.ndarray,list)):
        locs.extend(s_tail)
        tags.extend(['<ARG1>']*len(s_tail))
        locs.extend(e_tail)
        tags.extend(['</ARG1>']*len(e_tail))
    else:
        locs.append(s_tail)
        tags.append('<ARG1>')
        locs.append(e_tail)
        tags.append('</ARG1>')
        
    if isinstance(s_sig,(np.ndarray,list)):
        locs.extend(s_sig)
        tags.extend([f'<SIG{i}>' for i in range(len(s_sig))])
        locs.extend(e_sig)
        tags.extend([f'</SIG{i}>' for i in range(len(e_sig))])
    else:
        locs.append(s_sig)
        tags.append(f'<SIG0>')
        locs.append(e_sig)
        tags.append(f'</SIG0>')

    return sorted(zip(locs, tags))


def get_del_spl_list(s_del=[], e_del=[], e_split=[]):
    locs, tags = [], []
    if isinstance(s_del,(np.ndarray,list)):
        locs.extend(s_del)
        tags.extend([f'<DELETE>' for i in range(len(s_del))])
        locs.extend(e_del)
        tags.extend([f'</DELETE>' for i in range(len(e_del))])
    else:
        locs.append(s_del)
        tags.append('<DELETE>')
        locs.append(e_del)
        tags.append('</DELETE>')

    if isinstance(e_split,(np.ndarray,list)):
        locs.extend(e_split)
        tags.extend([f'</SPLIT>' for i in range(len(e_split))])
    else:
        locs.append(e_split)
        tags.append('</SPLIT>')

    return sorted(zip(locs, tags))


class Subtask2Annotations(object):
    def __init__(self, ref_df, root_ann_folder, folder_name, annotators, add_cleanedtext=True) -> None:
        # Option to include data cleaning layer or not
        self.add_cleanedtext = add_cleanedtext

        # For file path locations
        self.ref_df = ref_df
        self.suffix = folder_name.split('_')[1][:-4]
        self.ann_folder = os.path.join(root_ann_folder, folder_name)
        self.root_ann_folder = root_ann_folder
        self.folder_name = folder_name
        self.annotators = annotators
        self.check_annotators()

        # For general annotations
        self.annotations={}
        self.stored_relations={}
        self.spanids_w_rel = []
        self.global_errors = {
            'missing': [], # spans with no links
            'no_label': [], # spans with no labels
            'backwards': [], # e->c by mistake
            'no_cause': [], # s->e found, but no c->e main relation
            'effects_only': [], # nonsense e->e relation
            'no_effects': [], # nonsense c/s->c/s relation
        }
        self.span2sentid = {}
        self.list_of_dc_actions = []
        self.sentid2sentences = {}

        # For agreement scores
        self.metrics = {}
        self.reset_metrics()
    

    def check_annotators(self):
        # remove annotators if they don't have a JSON file
        for i,ann_name in enumerate(self.annotators):
            if not os.path.exists(os.path.join(self.ann_folder, f"{ann_name}.json")):
                self.annotators.remove(ann_name)


    def reset_metrics(self):
        self.metrics = {
            'compareAnns': [],
            'NumRels': [],
            'ExactMatch':{'Effect':[],'Cause':[],'Signal':[],'All':[]},
            'OneSideBound':{'Effect':[],'Cause':[],'Signal':[],'All':[]},
            'TokenOverlap':{'Effect':[],'Cause':[],'Signal':[],'All':[]},
            'Count':{'Effect':[],'Cause':[],'Signal':[],'All':[]},
            'KAlpha':{'Effect':[],'Cause':[],'Signal':[],'All':[]},
            'KCount':{'Effect':[],'Cause':[],'Signal':[],'All':[]}
        }


    def format_metrics(self):
        subset = os.path.splitext(self.folder_name)[0]
        metrics_df = []
        annotators = []
        for k,v in self.metrics.items():
            if type(v)==list:
                if k!='compareAnns':
                    metrics_df.append([subset, k, 'All', np.mean(v)])
                else:
                    annotators = list(set(sum(v, ())))
            else:
                for kk,vv in v.items():
                    metrics_df.append([subset, k, kk, np.mean(vv)])
        metrics_df = pd.DataFrame(
            metrics_df, 
            columns=['Subsample','Metric','Span Type','Agreement']
            )
        metrics_df['Annotators'] = ','.join(annotators)
        xx_roundxx = os.path.basename(os.path.dirname(self.root_ann_folder)) # E.g. "07. Round5" 
        metrics_df['Round'] = re.search(r'(?<=Round)[^.]*',xx_roundxx)[0] # E.g. "5"
        return metrics_df


    def calculate_pico(self):

        dummy_dict = {
            'Signal': '',
            'Signal_loc': (-1, -1),
            'Effect': '',
            'Effect_loc': (-1, -1),
            'Cause': '',
            'Cause_loc': (-1, -1)
        }

        for (annotator1, annotator2) in list(combinations(self.annotators,2)):

            to_compare = []
            for_kalpha = []
            num_rels = []
            exactmatch = {'Effect':[],'Cause':[],'Signal':[],'All':[]}
            onesidebound = {'Effect':[],'Cause':[],'Signal':[],'All':[]}
            tokenoverlap = {'Effect':[],'Cause':[],'Signal':[],'All':[]}

            for sentid in set(self.span2sentid.values()): #-set([0]): 
                if self.suffix in ['s01','s02'] and sentid==0:
                    # EXCLUDE FIRST EXAMPLE IS A DEMO 
                    continue
                anns1, anns2 = [], []
                for k,v in self.stored_relations.items():
                    if self.span2sentid[k]==sentid:
                        if v['Annotator']==annotator1:
                            anns1.append(v)
                        elif v['Annotator']==annotator2:
                            anns2.append(v)

                if len(anns1)==len(anns2):
                    num_rels.append(1)
                else:
                    num_rels.append(0)
                    if len(anns1)<len(anns2):
                        dummy_dict['Annotator']=annotator1
                        anns1.extend([dummy_dict]*(len(anns2)-len(anns1)))
                    else:
                        dummy_dict['Annotator']=annotator2
                        anns2.extend([dummy_dict]*(len(anns1)-len(anns2)))
                
                for a1, ann1 in enumerate(anns1):
                    for a2, ann2 in enumerate(anns2):
                        
                        fka = []
                        for k in ['Effect','Cause','Signal']:
                            
                            if k in ann1.keys():
                                begin, end = ann1[f'{k}_loc']
                                if begin==end==-1:
                                    fka.append([k,begin,end,ann1['Annotator']])
                                else:
                                    text = self.annotations[sentid]['retrieve_info']['text']
                                    buffer = self.annotations[sentid]['retrieve_info']['begin']
                                    start = len(text[:begin-buffer].split(' '))-1
                                    stop = start+len(text[begin-buffer:end-buffer].split(' '))
                                    fka.append([k,start,stop,ann1['Annotator']])
                            if k in ann2.keys():
                                begin, end = ann2[f'{k}_loc']
                                if begin==end==-1:
                                    fka.append([k,begin,end,ann1['Annotator']])
                                else:
                                    text = self.annotations[sentid]['retrieve_info']['text']
                                    buffer = self.annotations[sentid]['retrieve_info']['begin']
                                    start = len(text[:begin-buffer].split(' '))-1
                                    stop = start+len(text[begin-buffer:end-buffer].split(' '))
                                    fka.append([k,start,stop,ann2['Annotator']])
                            
                            if k in ann1.keys() and k in ann2.keys():
                                if ann1[f'{k}_loc']==ann2[f'{k}_loc']:
                                    exactmatch[k].append(1)
                                else:
                                    exactmatch[k].append(0)

                                if ann1[f'{k}_loc'][0]==ann2[f'{k}_loc'][0] or ann1[f'{k}_loc'][1]==ann2[f'{k}_loc'][1]:
                                    onesidebound[k].append(1)
                                else:
                                    onesidebound[k].append(0)

                                if any(i in range(ann1[f'{k}_loc'][0],ann1[f'{k}_loc'][1]) for i in range(ann2[f'{k}_loc'][0],ann2[f'{k}_loc'][1])):
                                    tokenoverlap[k].append(1)
                                else:
                                    tokenoverlap[k].append(0)

                            elif k in ann1.keys() or k in ann2.keys():
                                exactmatch[k].append(0)
                                onesidebound[k].append(0)
                                tokenoverlap[k].append(0)
                                
                            else: # Both missing, E.g. Signal
                                exactmatch[k].append(1)
                                onesidebound[k].append(1)
                                tokenoverlap[k].append(1)
                        
                        to_compare.append((sentid,a1,a2))
                        for_kalpha.append(fka)
                        exactmatch['All'].append(int(all([True if exactmatch[k][-1]==1 else False for k in ['Effect','Cause','Signal']])))
                        onesidebound['All'].append(int(all([True if onesidebound[k][-1]==1 else False for k in ['Effect','Cause','Signal']])))
                        tokenoverlap['All'].append(int(all([True if tokenoverlap[k][-1]==1 else False for k in ['Effect','Cause','Signal']])))

            keep_best_combinations_only(num_rels, to_compare, for_kalpha, exactmatch, onesidebound, tokenoverlap)

            self.metrics['compareAnns'].append((annotator1,annotator2))
            self.metrics['NumRels'].append(np.mean(num_rels))
            for k,v in exactmatch.items():
                self.metrics['ExactMatch'][k].append(np.mean(v))
                self.metrics['Count'][k].append(len(v))
            for k,v in onesidebound.items():
                self.metrics['OneSideBound'][k].append(np.mean(v))
            for k,v in tokenoverlap.items():
                self.metrics['TokenOverlap'][k].append(np.mean(v))
            
            overall = {'Effect':[],'Cause':[],'Signal':[],'All':[]}
            for i,anns in enumerate(for_kalpha):
                sentid=to_compare[i][0]
                text = self.annotations[sentid]['retrieve_info']['text']
                splitted = text.rstrip().split(' ')
                doc_length = len(splitted)
                all_df = pd.DataFrame(anns,columns=['tag', 'start', 'end', 'annotator'])
                all_df = all_df[all_df['end']>=0]
                all_df['end'] = all_df['end']-1 # Need to be inclusive
                all_df["focus"] = "result" # This is just a placeholder
                all_df['entity'] = all_df[['start', 'end', 'tag']].apply(
                    lambda row: (row.start, row.end, row.tag), axis=1)
                result = get_result(all_df, doc_length, overlap=True, empty_tag="empty")
            #     print(f"{sentid} -> Krippendorff's Alpha is : " + str(result))
                for k,(value,counts) in result['result'].items():
                    overall[k].extend([value]*counts)
                    overall['All'].extend([value]*counts)

            for k,v in overall.items():
                self.metrics['KAlpha'][k].append(np.mean(v))
                self.metrics['KCount'][k].append(len(v))


    def parse(self):

        ##### Obtain annotations #####
        for i,ann_name in enumerate(self.annotators):
            ann_file=open_json(os.path.join(self.ann_folder, f"{ann_name}.json"), dict)
            self.format_causality_annotation(ann_file, ann_name, constant=10000*i)
            self.span2sentid = self.get_span2sentid()
            self.format_into_relations(ann_file, constant=10000*i)
            if self.add_cleanedtext:
                self.clean_text()
        self.span2sentid = self.get_span2sentid()
        
        ##### Final Checks #####
        # Check if annotators linked properly
        # 1. All spans should be linked to something
        for sentid, infos in self.annotations.items():
            for k,v in infos['spans_info'].items():
                if k not in self.spanids_w_rel:
                    v_out = v.copy()
                    v_out['sentid'] = sentid
                    v_out['spanid'] = k
                    self.global_errors['missing'].append(v_out)
        # 2. Effect: Should always have a Cause link (not just signal!)
        for spanid, num_links in Counter(self.spanids_w_rel).items():
            if spanid in self.stored_relations.keys(): # If is Effect spanid
                v = self.stored_relations[spanid]
                if 'Cause' not in v.keys():
                    v_out = v.copy()
                    v_out['spanid'] = spanid
                    self.global_errors['no_cause'].append(v_out)


    def get_ces_output(self):
        output = []
        if self.add_cleanedtext:
            output_cols = ['Index', 'CleanedText', 'Cause', 'Effect', 'Signal', 'Annotator']
        else:
            output_cols = ['Index', 'Cause', 'Effect', 'Signal', 'Annotator']
        for k,v in self.stored_relations.items():
            # Index, E.g. 58774 --> 1
            row = [self.span2sentid[k]+1]
            # Remaining Columns
            for c in output_cols[1:]:
                if c=='Signal':
                    all_signals = [vv for kk,vv in v.items() if c in kk and 'loc' not in kk]
                    row.append('; '.join(all_signals))
                elif c in v.keys():
                    row.append(v[c])
                else:
                    row.append(None)
            output.append(row)
        return pd.DataFrame(output, columns=output_cols)


    def get_output(self, source = None):
        # Generate Examples
        corpus = 'cnc'
        cols = ['corpus','doc_id','sent_id','eg_id','index','text','text_w_pairs','seq_label','pair_label']
        data = []
        sentid_counter = defaultdict(int)
        sentid_subsplits = {}
        for k,v in self.stored_relations.items():

            sentid = self.span2sentid[k] # 0 (Note we add 1 to all sentid later)
            text = v['CleanedText'] # TRS condemns arrests March 11 ...
            text_w_pairs = v['text_w_pairs']
            slice_df = self.ref_df[(self.ref_df['source']==source) & (self.ref_df['Index']==sentid+1)]
            sentid_global = str(slice_df['index'].iloc[0]) # train_05_299 _0
            sentid_anns = str(sentid+1) # 4
            if sentid_anns in sentid_subsplits:
                if text in sentid_subsplits[sentid_anns]:
                    subid = sentid_subsplits[sentid_anns][text]
                else:
                    subid = len(sentid_subsplits[sentid_anns])
                    sentid_subsplits[sentid_anns][text] = subid
            else:
                subid = 0
                sentid_subsplits[sentid_anns] = {text:subid}
            sentid_anns=sentid_anns+'_'+str(subid)
            num_eg_for_this_sentid = sentid_counter[sentid_anns]
            identifiers = [corpus,sentid_global,sentid_anns,str(num_eg_for_this_sentid)]
            unique_index = '_'.join(identifiers)
            seq_label = pair_label = 1
            data.append(
                identifiers+[
                    unique_index,
                    text,
                    text_w_pairs,
                    seq_label,
                    pair_label
                ]
            )
            sentid_counter[sentid_anns]+=1

        data = pd.DataFrame(data, columns=cols)
        data['sent_id'] = data['sent_id'].astype(str)
        data['seq_label'] = data.groupby(['corpus','doc_id','sent_id'])['seq_label'].transform('max')

        # Add substrings (sentences) without annotations as "Not Causal"
        ref_df2 =  []
        for sentid, sentences in self.sentid2sentences.items():
            slice_df = self.ref_df[(self.ref_df['source']==source) & (self.ref_df['Index']==sentid+1)]
            sentid_global = str(slice_df['index'].iloc[0]) # train_05_299 _0
            for sent in sentences:
                ref_df2.append([sentid_global, str(sentid+1), sent.strip()])
        ref_df2 = pd.DataFrame(ref_df2, columns=['doc_id','sent_id','text'])
        ref_df2 = ref_df2.merge(data[['doc_id','text','seq_label']].drop_duplicates(), how='left', on=['doc_id','text'])
        not_causal_df = ref_df2[ref_df2['seq_label'].fillna(0)==0]
        if len(not_causal_df)>0:
            for i, row in not_causal_df.iterrows():
                identifiers = [corpus,row['doc_id'],str(row['sent_id']),str(0)]
                unique_index = '_'.join(identifiers)
                data.loc[len(data)] = identifiers+[
                    unique_index,
                    row['text'].strip(),
                    '',0,0]
        # Add unannotated examples as "Not Causal"
        not_causal_df = self.ref_df[(self.ref_df['source']==source) & (~self.ref_df['index'].isin(data['doc_id'].unique()))]
        if len(not_causal_df)>0:
            for i, row in not_causal_df.iterrows():
                identifiers = [corpus,row['index'],str(row['Index']),str(0)]
                unique_index = '_'.join(identifiers)
                data.loc[len(data)] = identifiers+[
                    unique_index,
                    row['text'].strip(),
                    '',0,0]
        
        return data


    def prepare_report(self, sub, split_by_annotator=False):
        
        ##### Reporting #####
        report = format_report(self.annotations, self.global_errors, self.span2sentid)
        report['Index']+=1

        ces_output = self.get_ces_output()
        ces_output['source']=self.folder_name
        ces_output.to_csv(os.path.join(self.ann_folder, "reviewed_all_{0}{1:02d}.csv".format(midfix, sub)), index=False, encoding='utf-8-sig')
        
        if split_by_annotator:
            # Annotator Split
            fn = os.path.splitext(os.path.basename(self.ann_folder))[0]
            for a in ces_output['Annotator'].unique():
                xls_path = os.path.join(self.ann_folder, f"{fn}_{a}.xlsx")
                with ExcelWriter(xls_path) as writer:
                    report[report['Annotator']==a].to_excel(writer,'errors',index=False)
                    ces_output[ces_output['Annotator']==a].to_excel(writer,'success',index=False)

        if len(report)>0:
            report.to_csv(os.path.join(self.ann_folder, "error_report_{0}{1:02d}.csv".format(midfix, sub)), index=False, encoding='utf-8-sig')
            print(f'failed: {self.folder_name}')
            return 0 # failed
        else:
            output = self.get_output(os.path.splitext(self.folder_name)[0])
            output.to_csv(os.path.join(self.ann_folder, f"reviewed_all.csv"), index=False, encoding='utf-8-sig')
            return 1 # pass


    def clean(self, text):
        return text


    def get_span2sentid(self, span2sentid={}):
        for sentid, infos in self.annotations.items():
            for s in infos['spans_info'].keys():
                span2sentid[s]=sentid
        return span2sentid


    def clean_text(self):
        # Convert DataCleaning Actions to SentID
        current_sentid = 0 # 0-indexed
        max_sentid = max(self.annotations.keys())
        dc2sentid = {}
        for i,dc_dict in enumerate(self.list_of_dc_actions):
            
            while current_sentid<max_sentid:
                tmp_d = self.annotations[current_sentid]['retrieve_info']
                sent_begin, sent_end = tmp_d['begin'],tmp_d['end']
                if sent_end>=dc_dict['end']:
                    break 
                else:
                    current_sentid+=1
            dc2sentid[i]=current_sentid

        sent2dcid = defaultdict(list)
        for k,v in dc2sentid.items():
            sent2dcid[v].append(k)
        sent2dcid = dict(sent2dcid)

        # Update CleanedText of each span example
        for spanid, v in self.stored_relations.items():
            sentid = self.span2sentid[spanid]
            tmp_d = self.annotations[sentid]['retrieve_info']

            text_w_pairs = self.annotations[sentid]['retrieve_info']['text']
            added_t = 0
            accounting = self.annotations[sentid]['retrieve_info']['begin']
            sorted_s_e_t_list = get_s_e_t_list(
                s_head=v['Cause_loc'][0], 
                e_head=v['Cause_loc'][1], 
                s_tail=v['Effect_loc'][0], 
                e_tail=v['Effect_loc'][1],
                s_sig=[vv[0] for kk,vv in v.items() if kk[:6]=='Signal' and kk[-3:]=='loc'],
                e_sig=[vv[1] for kk,vv in v.items() if kk[:6]=='Signal' and kk[-3:]=='loc']
            )
            if sentid in sent2dcid:
                s_del, e_del, e_split = [], [], []
                for dcid in sent2dcid[sentid]:
                    if self.list_of_dc_actions[dcid]['DataCleaning']=='Delete':
                        pass
                        # s_del.append(self.list_of_dc_actions[dcid]['begin'] if 'begin' in self.list_of_dc_actions[dcid] else 0)
                        # e_del.append(self.list_of_dc_actions[dcid]['end'])
                    else: #['DataCleaning']===='Split'
                        e_split.append(self.list_of_dc_actions[dcid]['end'])
                sorted_del_spl_list = get_del_spl_list(s_del, e_del, e_split)
                sorted_tags_list=sorted_s_e_t_list+sorted_del_spl_list
            else:
                sorted_tags_list=sorted_s_e_t_list

            for loc,tag in sorted(sorted_tags_list):
                loc+=added_t-accounting
                text_w_pairs = text_w_pairs[:loc]+tag+text_w_pairs[loc:]
                added_t+=len(tag)

            text_w_pairs = re.sub("<DELETE>.*?</DELETE>","",text_w_pairs.strip())
            
            # Store texts so that substrings (sentences) can be retrieved
            if sentid not in self.sentid2sentences:
                self.sentid2sentences[sentid] = [re.sub("</?.*?>","",s).strip() for s in text_w_pairs.split('</SPLIT>')]
            
            text_w_pairs = [substring.strip() for substring in text_w_pairs.split('</SPLIT>') if ('<ARG0>' in substring) and ('<ARG1>' in substring)]
            assert(len(text_w_pairs)==1)
            text_w_pairs = text_w_pairs[0]
            self.stored_relations[spanid]['text_w_pairs'] = text_w_pairs
            self.stored_relations[spanid]['CleanedText'] = re.sub("</?.*?>","",text_w_pairs)


    def format_into_relations(self, ann_file, constant=0):
        
        ##### Get Relations + Error reporting #####
        layer = 'CausalLabel'
        for rel in ann_file['_views']['_InitialView'][layer]:

            e_spanid, cs_spanid = rel['Dependent'], rel['Governor']
            e_spanid += constant
            cs_spanid += constant
            sentid = self.span2sentid[e_spanid]
            self.spanids_w_rel.append(e_spanid)
            self.spanids_w_rel.append(cs_spanid)
            
            effect = self.annotations[sentid]['spans_info'][e_spanid]
            cause_or_sig = self.annotations[sentid]['spans_info'][cs_spanid]
            
            if cause_or_sig['label'] == effect['label'] == 'Effect':
                self.global_errors['effects_only'].append([rel,effect,cause_or_sig])
                continue # go next iter
            
            if cause_or_sig['label'] == 'Missing' or effect['label'] == 'Missing':
                if cause_or_sig['label'] == 'Missing':
                    v_out = cause_or_sig.copy()
                    v_out['spanid'] = cs_spanid
                    v_out['sentid'] = sentid
                    self.global_errors['no_label'].append(v_out)
                if effect['label'] == 'Missing' == 'Missing':
                    v_out = effect.copy()
                    v_out['spanid'] = e_spanid
                    v_out['sentid'] = sentid
                    self.global_errors['no_label'].append(v_out)
                continue # go next iter
            
            if cause_or_sig['label'] == 'Effect':
                # possibly swapped by mistake
                cs_spanid, e_spanid = rel['Dependent'], rel['Governor']
                e_spanid += constant
                cs_spanid += constant
                sentid = self.span2sentid[e_spanid]
                effect = self.annotations[sentid]['spans_info'][e_spanid]
                cause_or_sig = self.annotations[sentid]['spans_info'][cs_spanid]
                swap_status = True
            else:
                swap_status = False
            
            if 'CausalLabel' in rel.keys():
                rel_label = rel['CausalLabel']
            else:
                rel_label = None # annotator did not specify
            
            ann_name = effect['annotator']
            cs_label = cause_or_sig['label']
            
            effect['spanid']=e_spanid
            cause_or_sig['spanid']=cs_spanid

            if effect['label'] != 'Effect':
                self.global_errors['no_effects'].append([rel,cause_or_sig,effect])
                continue # go next iter
            
            text = self.annotations[sentid]['retrieve_info']['text']
            buffer = self.annotations[sentid]['retrieve_info']['begin']
            effect_text = text[effect['begin']-buffer:effect['end']-buffer]
            cause_or_sig_text = text[cause_or_sig['begin']-buffer:cause_or_sig['end']-buffer]

            if e_spanid not in self.stored_relations.keys():
                self.stored_relations[e_spanid]={
                    'Effect':effect_text, 
                    'Relation':rel_label, 
                    'Annotator': ann_name,
                    'Effect_loc': (effect['begin'],effect['end'])
                    }
                if self.add_cleanedtext:
                    self.stored_relations[e_spanid]['CleanedText'] = text.strip()

            if cs_label not in self.stored_relations[e_spanid].keys():
                self.stored_relations[e_spanid][cs_label]=cause_or_sig_text
                self.stored_relations[e_spanid][cs_label+'_loc']=(cause_or_sig['begin'],cause_or_sig['end'])
            else: # allow for multiple signal spans
                all_keys = list(self.stored_relations[e_spanid].keys())
                current_counter = sum([1 for i in all_keys if cs_label in i])
                self.stored_relations[e_spanid][cs_label+f'_{current_counter+1}']=cause_or_sig_text
                self.stored_relations[e_spanid][cs_label+f'_{current_counter+1}'+'_loc']=(cause_or_sig['begin'],cause_or_sig['end'])
            
            if swap_status:
                v_out = self.stored_relations[e_spanid].copy()
                v_out['spanid'] = e_spanid
                self.global_errors['backwards'].append(v_out)


    def format_causality_annotation(self, ann_file, ann_name='Unspecified', constant=0):

        ##### Get Texts #####
        sofa=str(12)
        text=self.clean(ann_file['_referenced_fss'][sofa]['sofaString'])
        
        sent2locid = {-1:-2}
        for counter, sent in enumerate(text.split('\r\n')):
            sent2locid[counter]=sent2locid[counter-1]+len(sent)+2
            # Initialise
            sent_begin, sent_end = max(0,sent2locid[counter-1]), min(sent2locid[counter]+2,len(text))
            self.annotations[counter]={
                'spans_info': {},
                'retrieve_info': {
                    'sofa': sofa,
                    'begin': sent_begin,
                    'end': sent_end,
                    'text': text[sent_begin:sent_end]
                }
            }

        ##### Get DataCleaning Annotations #####
        if self.add_cleanedtext:
            layer = 'DataCleaning'
            if layer in ann_file['_views']['_InitialView'].keys():
                self.list_of_dc_actions = list(ann_file['_views']['_InitialView'][layer])
        
        ##### Get Marked Spans #####
        layer = 'CauseEffectSignal'
        list_of_span_ids = ann_file['_views']['_InitialView'][layer]
        max_span_id = max([x for x in list_of_span_ids if isinstance(x, (int,float))])

        for span_id in list_of_span_ids:
            
            # corrupted for some
            if not isinstance(span_id, (int,float)):
                a_infos = span_id
                span_id = max_span_id
                max_span_id += 1
            else:
                a_infos = ann_file['_referenced_fss'][str(span_id)]
                
            # process
            begin, end = 0, len(text)
            if 'begin' in a_infos.keys():
                begin = a_infos['begin']
            if 'end' in a_infos.keys():
                end = a_infos['end']

            sents_needed = find_sents_needed(sent2locid, begin, end)
            if len(sents_needed)>1:
                raise ValueError('We should be annotating within sentences only!')
            else:
                sentid = sents_needed[0]
                sent_begin, sent_end = max(0,sent2locid[sentid-1]), min(sent2locid[sentid]+2,len(text))
            
            if 'CES' in a_infos.keys():
                label = a_infos['CES']
            else:
                label = 'Missing'
            
            span_dict = {
                    'label': label,
                    'begin': begin,
                    'end': end,
                    'annotator': ann_name,
            }
            
            span_id += constant
            
            if sentid in self.annotations.keys():
                assert(self.annotations[sentid]['retrieve_info']['sofa']==sofa)
                assert(self.annotations[sentid]['retrieve_info']['begin']==sent_begin)
                assert(self.annotations[sentid]['retrieve_info']['end']==sent_end)
                self.annotations[sentid]['spans_info'][span_id]=span_dict
            else:
                # initialise
                raise ValueError('All sentences should be initialised at sent2locid creation step.')


def join_all_ces(ref_df, root_ann_folder, samples):
    output_df = pd.DataFrame()
    for sub in samples:
        folder_name = "subtask2_{0}{1:02d}.txt".format(midfix, sub)
        ann_folder = os.path.join(root_ann_folder, folder_name)
        output = pd.read_csv(os.path.join(ann_folder, "reviewed_all_{0}{1:02d}.csv".format(midfix, sub)))
        output_df = pd.concat([output_df, output], axis=0)
    output_df = output_df.reset_index(drop=True)
    output_df['source'] = output_df['source'].apply(lambda x: os.path.splitext(x)[0])
    final_df = ref_df.merge(output_df, how='right', on=['source','Index'])
    final_df.to_csv(
        os.path.join(root_ann_folder, "reviewed_all_{0}{1:02d}_s{2:02d}.csv".format(midfix, min(samples), sub)), 
        index=False, encoding='utf-8-sig'
        )

def join_all(root_ann_folder, samples):
    output_df = pd.DataFrame()
    for sub in samples:
        folder_name = "subtask2_{0}{1:02d}.txt".format(midfix, sub)
        ann_folder = os.path.join(root_ann_folder, folder_name)
        output = pd.read_csv(os.path.join(ann_folder, f"reviewed_all.csv"))
        output_df = pd.concat([output_df, output], axis=0)
    
    output_df.reset_index(drop=True).to_csv(
        os.path.join(root_ann_folder, f"reviewed_all.csv"), 
        index=False, encoding='utf-8-sig'
        )


if __name__ == "__main__":

    """
    Command Line:
        conda activate pytorch
        cd D:\61 Challenges\2022_CASE_\CausalNewsCorpus\CausalNewsCorpus\curation
        python subtask2.py
    """
    # Change per run: 
    samples = list(range(1,40+1)) #list(range(1,4+1))
    root_ann_folder = r"D:\61 Challenges\2022_CASE_\WebAnno\reviewing_annotations\Subtask2\15. Round13\curation"
    
    # Do not touch the remaining:
    ref_df = get_ref_df(root_ann_folder)
    passed = 0
    for sub in tqdm(samples):
        self = Subtask2Annotations(
            ref_df = ref_df,
            root_ann_folder = root_ann_folder, 
            folder_name = "subtask2_{0}{1:02d}.txt".format(midfix, sub),
            annotators = ['CURATION_USER'],
            add_cleanedtext = True
            )
        self.parse()
        passed+=self.prepare_report(sub)
    print(f'Proportion of passed subsamples: {passed/len(samples)}')

    join_all_ces(ref_df, root_ann_folder, samples)
    join_all(root_ann_folder, samples)


