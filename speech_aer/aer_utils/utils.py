import os
import json
import itertools
import string
import numpy as np
from collections import Counter, defaultdict
from aer_utils.small_utils import *


def get_word2word_contrib(contributions, setting, speech_src_dict, speech_tgt_dict=None, tgt_sentence=None):

    """
    Function to obtain word2word contributions instead of token2token
    Inputs:
        contributions: numpy tensor with the contributions, of shape [output_seq_len, input_seq_len].
        setting: Speech to Speech (s2s) or Speech to Text (s2t).
        speech_src_dict: dictionary from the Speech Gold Alignment dataset in the source language.
        speech_tgt_dict: dictionary from the Speech Gold Alignment dataset in the target language (only for s2s).
        tgt_sentence: tokenized subwords (only for s2t).
    """
    contributions = contributions.detach().cpu().numpy()

    if setting == 's2s':
        assert speech_tgt_dict
    if setting == 's2t':
        assert tgt_sentence

    if isinstance(tgt_sentence, str):
        tgt_sentence = tgt_sentence.split(' ')
    
    contributions, words_in, words_out = contrib_tok2words(
        contributions,
        tokens_in=None,
        tokens_out=tgt_sentence,
        setting=setting,
        speech_src_dict=speech_src_dict,
        speech_tgt_dict=speech_tgt_dict
        )

    source_sentence_ =  words_in
    tgt_sentence_ = words_out

    return contributions, source_sentence_, tgt_sentence_
    


def get_alignments(contributions, src_words:list, tgt_words:list):
    a_argmax = np.argmax(contributions, -1)
    contributions_word_word_hard = np.zeros(contributions.shape)

    for i, j in enumerate(a_argmax):
        contributions_word_word_hard[i][j] = 1

    contributions_word_word_hard = duplicate_rows_cols(contributions_word_word_hard, src_words, tgt_words)

    tgt_words = separate_words(tgt_words) 

    # Source will never have punctuation
    alignments_string = ''
    for row_idx, row in enumerate(contributions_word_word_hard):
        if tgt_words[row_idx] not in string.punctuation:
            column_index = np.nonzero(row)[0]
            for col_idx in column_index:
                alignments_string = alignments_string + str(col_idx+1) + '-' +  str(row_idx+1) + ' '
    alignments_string = alignments_string + '\n'

    return contributions_word_word_hard, alignments_string


def calculate_saer(path_hypo_file, path_gold_alignment, translation_direction):
    """
    Calculate Word-SAER (alignment error rate), Precision and Recall for constructed alignments in the layer-level.
    """

    sure, possible = [], []
    with open(path_gold_alignment, 'r') as f:
        for line in f:
            sure.append(set())
            possible.append(set())

            for alignment_string in line.split():

                sure_alignment = True if '-' in alignment_string else False
                if translation_direction == 'en-de':
                    alignment_tuple = parse_single_alignment(alignment_string, reverse=True, one_indexed=True)
                else:
                    alignment_tuple = parse_single_alignment(alignment_string, one_indexed=True)

                if sure_alignment:
                    sure[-1].add(alignment_tuple)
                possible[-1].add(alignment_tuple)


    assert len(sure) == len(possible)

    metrics = defaultdict(list)

    hypothesis = []

    with open(path_hypo_file) as f:
        for line in f:
            hypothesis.append(set())

            for alignment_string in line.split():
                alignment_tuple = parse_single_alignment(alignment_string, one_indexed=True)
                hypothesis[-1].add(alignment_tuple)

    sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 4 * [0.0]
    for S, P, A in itertools.zip_longest(sure, possible, hypothesis):

        sum_a += len(A)
        sum_s += len(S)
        sum_a_intersect_p += len(A.intersection(P))
        sum_a_intersect_s += len(A.intersection(S))

    precision = sum_a_intersect_p / sum_a
    recall = sum_a_intersect_s / sum_s
    aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['aer'].append(aer)
                

    return metrics
    

def calculate_twsaer(path_hypo_file, path_gold_alignment, translation_direction, setting, src_weights=None, tgt_weights=None):
    """
    Calculate Time-SAER (alignment error rate), Precision and Recall for constructed alignments in the layer-level.
    """

    sure, possible = [], []
    
    if src_weights == None or tgt_weights == None:
        src_weights, tgt_weights = build_weights_dictionary(translation_direction, setting)

    with open(path_gold_alignment, 'r') as f:
        for line in f:
            sure.append([])
            possible.append([])

            for alignment_string in line.split():

                sure_alignment = True if '-' in alignment_string else False
                if translation_direction == 'en-de':
                    alignment_tuple = parse_single_alignment(alignment_string, reverse=True, one_indexed=True)
                else:
                    alignment_tuple = parse_single_alignment(alignment_string, one_indexed=True)

                if sure_alignment:
                    sure[-1].append(alignment_tuple)
                possible[-1].append(alignment_tuple)


    assert len(sure) == len(possible)

    metrics = defaultdict(list)
    hypothesis = []

    with open(path_hypo_file) as f:
        for line in f:
            hypothesis.append([])

            for alignment_string in line.split():
                alignment_tuple = parse_single_alignment(alignment_string, one_indexed=True)
                hypothesis[-1].append(alignment_tuple)

    sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 4 * [0.0]
    for idx, (S, P, A) in enumerate(itertools.zip_longest(sure, possible, hypothesis)):
        
        sum_a += compute_alignment_area(A, idx, setting, src_weights, tgt_weights)
        sum_s += compute_alignment_area(S, idx, setting, src_weights, tgt_weights)
        sum_a_intersect_p += compute_alignment_area([x for x in A if x in P], idx, setting, src_weights, tgt_weights)
        sum_a_intersect_s += compute_alignment_area([x for x in A if x in S], idx, setting, src_weights, tgt_weights)

    precision = sum_a_intersect_p / sum_a
    recall = sum_a_intersect_s / sum_s
    aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['aer'].append(aer)
                

    return metrics

