"""
Script for calculating Krippendorff's Alpha between two or more annotators 
Codes from Emerging Welfare 
https://github.com/emerging-welfare/kAlpha

"""

from __future__ import print_function
import sys
import re
import pandas as pd
import argparse
import itertools


def sorted_nicely(l): # alphanumeric string sort
    # Copied from this post -> https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key[0])]
    return sorted(l, key=alphanum_key)

def organize_data(entities, doc_length, empty_tag):
    """
    Organize the annotations in an acceptable format for calculate_kalpha function.
    """

    out_list = []
    start_point = 0
    if len(entities) == 0:
        out_list.append([1, doc_length, empty_tag])
        return out_list

    for entity in entities:
        # ESP = Entity Start Point , EEP = Entity End Point
        ESP = entity[0]
        EEP = entity[1]
        if ESP > start_point:
            out_list.append([start_point, ESP - 1, empty_tag])

        elif ESP < start_point:
            # TODO: Too many of this error, investigate!
            print("Duplicates or Overlapping Tags")
            continue

        out_list.append([ESP, EEP, entity[-1]])
        start_point = EEP + 1

    last_point = out_list[-1][1]
    if last_point < doc_length:
        out_list.append([last_point + 1, doc_length, empty_tag])

    return out_list


def get_union(a, b):
    return max(a[1], b[1]) - min(a[0], b[0]) + 1

def get_intersect(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) + 1

def get_length(a):
    return a[1] - a[0] + 1

def encapsulates(a, b):
    if get_intersect(a, b) == get_length(b):
        return True

    return False

def get_metric(a, b):
    if a[2] == b[2]:
        return 0

    return 1

def calculate_kalpha(in_entities, annots, empty_tag):
    pairs = list(itertools.combinations(list(range(len(annots))), 2))
    observed_nom = 0
    observed_denom = 0
    expected_nom = 0
    expected_denom = 0
    empty_count = 0
    weight = 0
    if len(in_entities) == 0:
        return 0.0, 0

    for pair in pairs:
        entities1 = in_entities[pair[0]]
        entities2 = in_entities[pair[1]]
        if entities1 == entities2:
            for g in entities1:
                if g[2] != empty_tag:
                    observed_denom += 1
                    weight += 2 * get_length(g)

            continue

        for g in entities1:
            for h in entities2:
                intersect = get_intersect(g, h)
                if intersect > 0:
                    if g[2] != empty_tag and h[2] != empty_tag:
                        observed_nom += get_union(g, h) - intersect * (1 - get_metric(g, h))
                        observed_denom += 1
                        weight += get_length(g) + get_length(h)

                    elif g[2] != empty_tag and h[2] == empty_tag and encapsulates(h, g):
                        observed_nom += 2 * get_length(g)
                        observed_denom += 1
                        weight += get_length(g)

                    elif g[2] == empty_tag and h[2] != empty_tag and encapsulates(g, h):
                        observed_nom += 2 * get_length(h)
                        observed_denom += 1
                        weight += get_length(h)

                    elif g[2] == empty_tag and h[2] == empty_tag:
                        empty_count += intersect

    if observed_nom == 0:
        return 1.0, weight

    observed = float(observed_nom / observed_denom)
    entities = [i for x in in_entities
                for i in x if i[2] != empty_tag]
    expected = 0.0
    for g in entities:
        seenItself = False
        for h in entities:
            # TODO: See if this comparison is correct? Since two annotators can agree on some annotation,
            # g and h might come from different annotators but still be exactly same.
            if g == h and not seenItself:
                seenItself = True
                continue

            leng = get_length(g)
            lenh = get_length(h)
            expected_nom += leng * leng + lenh * lenh + leng * lenh * get_metric(g, h)
            expected_denom += leng + lenh

    if expected_denom != 0:
        expected = float(expected_nom / expected_denom)

    if expected == 0:
        return 0.0, weight

    return 1.0 - observed / expected, weight


def get_result(all_df, doc_length, overlap=False, empty_tag="empty"):
    annotators = all_df.annotator.unique().tolist()
    foliasets = all_df.focus.unique().tolist()
    kAlpha_sets = {}

    if not overlap:
        for foliaset in foliasets:
            all_entities = []
            for annot in annotators:
                entities = all_df[(all_df.annotator == annot) & (
                    all_df.focus == foliaset)].entity.tolist()
                entities = sorted(entities)
                entities = organize_data(entities, doc_length, empty_tag)
                all_entities.append(entities)

            kAlpha_sets[foliaset] = calculate_kalpha(all_entities, annotators, empty_tag)

    else:
        for foliaset in foliasets:
            curr_df = all_df[all_df.focus == foliaset]
            tags = curr_df.tag.unique()
            kAlpha_tags = {}
            for tag in tags:
                all_entities = []
                for annot in annotators:
                    entities = curr_df[(curr_df.annotator == annot) & (
                        curr_df.tag == tag)].entity.tolist()
                    entities = sorted(entities)
                    entities = organize_data(entities, doc_length, empty_tag)
                    all_entities.append(entities)

                kAlpha_tags[tag] = calculate_kalpha(all_entities, annotators, empty_tag)

            kAlpha_sets[foliaset] = kAlpha_tags

    return kAlpha_sets

def resolve_entity_discontinuity(entity_ids):
    """
    entity_ids is a sorted list of sentence and word ids of the entity.
    If tokens in the entity span are discontinuous, divides into multiple continuous parts.
    Returns a list of entities, each containing entity ids.
    """

    if len(entity_ids) == 1:
        return [entity_ids]

    entities = []
    idx = 0
    up_to_idx = 0
    while idx < len(entity_ids) - 1:
        curr_word_id = entity_ids[idx][1]
        next_word_id = entity_ids[idx+1][1]

        if curr_word_id + 1 != next_word_id: # if there is a discontinuity
            entities.append(entity_ids[up_to_idx:idx+1])
            up_to_idx = idx+1

        idx += 1

    entities.append(entity_ids[up_to_idx:])

    return entities


def get_ssp_and_doc_length(doc):
    sentence_starting_points = {}
    doc_length = 0
    for paragraph in doc.paragraphs():
        for sentence in paragraph.sentences():
            curr_para_sent_id = re.search(r'(p\.\d+\.s\.\d+)$', sentence.id).group(1)
            sentence_starting_points[curr_para_sent_id] = doc_length
            doc_length += len(sentence)

    return sentence_starting_points, doc_length
