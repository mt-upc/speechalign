import math
import string
import json
import numpy as np

# Merge contributions by token

def contrib_tok2words_partial(contributions, tokens, axis, reduction):
    from string import punctuation

    reduction_fs = {
        'avg': np.mean,
        'sum': np.sum
    }

    words = []
    w_contributions = []
    for counter, (tok, contrib) in enumerate(zip(tokens, contributions.T)):
        if tok.startswith('▁') or tok.startswith('__') or tok.startswith('<') or counter==0 or tok in punctuation:
            if tok.startswith('▁'):
                tok = tok[1:]
            words.append(tok)
            w_contributions.append([contrib])
        else:
            words[-1] += tok
            w_contributions[-1].append(contrib)

    reduction_f = reduction_fs[reduction]
    word_contrib = np.stack([reduction_f(np.stack(contrib, axis=axis), axis=axis) for contrib in w_contributions], axis=axis)

    return word_contrib, words

def contrib_tok2words_punctuation(contributions, tokens, axis, reduction):
    from string import punctuation
    merged_indices = []  # To keep track of the merged token indices

    for i in range(len(tokens) - 1):
        token1 = tokens[i]
        token2 = tokens[i + 1]

        if token1 == '' and token2 in punctuation:
            merged_token = token1 + token2
            merged_indices.append(i)  # Save the index to merge and average later
            tokens[i] = merged_token

    # Average the rows of the array corresponding to the merged indices
    for index in merged_indices:
        if reduction == 'avg':
            contributions[index] = np.mean(np.stack((contributions[index], contributions[index + 1])), axis=axis)
        elif reduction == 'sum':
            contributions[index] = np.sum(np.stack((contributions[index], contributions[index + 1])), axis=axis)

    # Remove the rows from the array that were averaged
    contributions = np.delete(contributions, merged_indices, axis=0)

    # Remove the merged tokens from the list
    tokens_list = [token for i, token in enumerate(tokens) if i not in merged_indices]

    return contributions, tokens_list


def contrib_speech_tok2words_partial(full_contributions, tokens, speech_src_dict, axis, reduction):
    while speech_src_dict[-1][-1] == None:
        speech_src_dict.pop(-1)
    total_duration = speech_src_dict[-1][-1]

    total_tokens = len(tokens)
    words = []
    word_tok_limits = []
    prev_s_time = -1
    for e in speech_src_dict:
        word, s_time, e_time = e
        if s_time == prev_s_time:
            words[-1] += ' '
            words[-1] += word
        elif s_time is not None and e_time:
            words.append(word)
            s_token = int(math.ceil(s_time*total_tokens/total_duration))
            e_token = int(math.floor(e_time*total_tokens/total_duration))
            word_tok_limits.append([s_token, e_token])
            prev_s_time = s_time

    if axis == 1:
        contributions = np.zeros((full_contributions.shape[0], len(word_tok_limits)))
    else:
        contributions = np.zeros((len(word_tok_limits), full_contributions.shape[1]))

    for i, (start, end) in enumerate(word_tok_limits):
        if reduction == 'sum':
            contributions[:, i] = np.sum(full_contributions[:, start:end], axis=axis)
        if reduction == 'avg':
            contributions[i, :] = np.mean(full_contributions[start:end, :], axis=axis)

    return contributions, words


def contrib_tok2words(contributions, tokens_in, tokens_out, setting, speech_src_dict, speech_tgt_dict=None):
    if not tokens_in:
        src_len = contributions.shape[1]
        tokens_in = [str(n) for n in range(src_len)]
    if not tokens_out:
        assert setting == 's2s'
        src_len = contributions.shape[1]
        tokens_in = [str(n) for n in range(src_len)]
        tgt_len = contributions.shape[0]
        tokens_out = [str(n) for n in range(tgt_len)]
    if setting == 's2t':
        word_contrib, words_in = contrib_speech_tok2words_partial(contributions, tokens_in, speech_src_dict, axis=1, reduction='sum')
        word_contrib, words_out = contrib_tok2words_partial(word_contrib.T, tokens_out, axis=0, reduction='avg')
        word_contrib, words_out = contrib_tok2words_punctuation(word_contrib, words_out, axis=0, reduction='avg')
    if setting == 's2s':
        word_contrib, words_in = contrib_speech_tok2words_partial(contributions, tokens_in, speech_src_dict, axis=1, reduction='sum')
        word_contrib, words_out = contrib_speech_tok2words_partial(word_contrib, tokens_out, speech_tgt_dict, axis=0, reduction='avg')
    return word_contrib, words_in, words_out

# Visualization

def separate_words(words_list):
    separated_words = []
    for word in words_list:
        separated_words.extend(word.split())
    return separated_words

# AER small utils

def duplicate_rows_cols(matrix, col_words, row_words):
    def split_word(word):
        return word.split() if ' ' in word else [word]

    row_counts = [len(split_word(word)) for word in row_words]
    col_counts = [len(split_word(word)) for word in col_words]

    new_rows = sum(row_counts)
    new_cols = sum(col_counts)

    new_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)

    row_idx = 0
    for row, count in enumerate(row_counts):
        for _ in range(count):
            col_idx = 0
            for col, col_word in enumerate(col_words):
                col_count = len(split_word(col_word))
                new_matrix[row_idx, col_idx:col_idx + col_count] = matrix[row, col]
                col_idx += col_count
            row_idx += 1

    return new_matrix

def parse_single_alignment(string, reverse=False, one_add=False, one_indexed=False):
    """
    Given an alignment (as a string such as "3-2" or "5p4"), return the index pair.
    """
    assert '-' in string or 'p' in string

    a, b = string.replace('p', '-').split('-')
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1
    
    if one_add:
        a = a + 1
        b = b + 1

    if reverse:
        a, b = b, a

    return a, b

def remove_punctuation(tensor, sentence):

    if len(tensor) != len(sentence):
        raise ValueError("The number of rows in the tensor should be equal to the number of elements in the word list.")

    punctuation_set = set(string.punctuation)
    indices_to_keep = [i for i, word in enumerate(sentence) if word not in punctuation_set]
    new_tensor = tensor[indices_to_keep]
    new_word_list = [word for word in sentence if word not in punctuation_set]

    return new_tensor, new_word_list

def build_weights_dictionary(translation_direction, setting):

    def get_weights(d, lang):
        weights_dict = {}
        prev_s_time = None
        for sentence in d:
            weights_dict[sentence] = {}
            if lang == 'en':
                sentence_dict = d[sentence]['lj_spk']
            elif lang == 'de':
                sentence_dict = d[sentence]['spk_thorsten']

            word_idx = 0
            for e in sentence_dict:
                word, s_time, e_time = e
                if s_time == prev_s_time:
                    weights_dict[sentence][word_idx] = e_time - s_time
                    prev_s_time = s_time
                    word_idx = word_idx + 1
                elif s_time is not None and e_time:
                    weights_dict[sentence][word_idx] = e_time - s_time
                    prev_s_time = s_time
                    word_idx = word_idx + 1
                elif s_time is None:
                    weights_dict[sentence][word_idx] = 0
                    word_idx = word_idx + 1
        return weights_dict

    path_en_dict = '/private/home/alastruey/speech_gold_alignment/dataset/english/en_dataset.json'
    path_de_dict = '/private/home/alastruey/speech_gold_alignment/dataset/german/de_dataset.json'

    with open(path_en_dict) as json_file:
        en_dict_all = json.load(json_file)
    with open(path_de_dict) as json_file:
        de_dict_all = json.load(json_file)
    
    if 'de' in translation_direction[:2]:
        src_dict, tgt_dict = de_dict_all, en_dict_all
    else:
        src_dict, tgt_dict = en_dict_all, de_dict_all
    
    src_weights = get_weights(src_dict, translation_direction[:2])

    if setting == 's2s':
        tgt_weights = get_weights(tgt_dict, translation_direction[-2:])
        return src_weights, tgt_weights

    return src_weights, None

def compute_alignment_area(alignments_set, sentence_idx, setting, src_weights, tgt_weights):

    alignment_src_weights = []
    alignment_tgt_weights = []

    for alignment in alignments_set:

        alignment_src_weights.append(src_weights[str(sentence_idx).zfill(3)][alignment[0]])
        if setting == 's2s':
            alignment_tgt_weights.append(tgt_weights[str(sentence_idx).zfill(3)][alignment[1]])
        else:
            alignment_tgt_weights.append(1)
        
    areas = [x * y for x, y in zip(alignment_src_weights, alignment_tgt_weights)]
    total_area = sum(areas)

    return total_area
