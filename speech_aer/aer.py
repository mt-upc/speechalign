import os
import json
import argparse
import torch
import json

from aer_utils.utils import get_word2word_contrib, get_alignments, calculate_saer, remove_punctuation, calculate_twsaer



def calculate_aercalculate_aer(test_set_dir, path_to_contribs, path_to_alignment_hyp, setting, translation_direction, path_to_tokenized_targets):

    with open(f'{test_set_dir}/dataset/english/en_dataset.json') as json_file:
        en_dict_all = json.load(json_file)

    with open(f'{test_set_dir}/dataset/german/de_dataset.json') as json_file:
        de_dict_all = json.load(json_file)

    if path_to_tokenized_targets:
        with open(f'{path_to_tokenized_targets}') as txt_file:
            all_tgt_tokens = txt_file.readlines()


    with open(path_to_alignment_hyp, 'w') as f:
        for i in range(len(all_tgt_tokens)):
        
            contrib_idx = str(i).zfill(3)
            contributions = torch.load(path_to_contribs+contrib_idx+'.pt')
            if 'en' in translation_direction[:2]:
                speech_src_dict = en_dict_all[contrib_idx]['lj_spk']
                speech_tgt_dict = de_dict_all[contrib_idx]['spk_thorsten']
            else:
                speech_src_dict = de_dict_all[contrib_idx]['spk_thorsten']
                speech_tgt_dict = en_dict_all[contrib_idx]['lj_spk']
            
            if setting == 's2t':
                tgt_tokens = all_tgt_tokens[int(contrib_idx)]
                tgt_tokens = tgt_tokens.replace('\n', '').split(' ') 
                assert tgt_tokens[-1] != '</s>', 'Remove special tokens and their corresponding contributions.'
                contributions, source_sentence_, tgt_sentence_ = get_word2word_contrib(contributions, setting, speech_src_dict, speech_tgt_dict=None, tgt_sentence=tgt_tokens)

            else:
                contributions, source_sentence_, tgt_sentence_ = get_word2word_contrib(contributions, setting, speech_src_dict, speech_tgt_dict=speech_tgt_dict, tgt_sentence=None)

            hard_alignment, alignments_string = get_alignments(contributions, source_sentence_, tgt_sentence_)
            f.write(alignments_string)

    path_gold_alignment = '/private/home/alastruey/speech_gold_alignment/dataset/alignment_no_punct.txt'
    print('SAER:')
    metrics = calculate_saer(path_to_alignment_hyp, path_gold_alignment, translation_direction)
    for metric in metrics:
        print(metric, ': ',  metrics[metric][0])
    print('TW-SAER:')
    metrics = calculate_twsaer(path_to_alignment_hyp, path_gold_alignment, translation_direction, setting)
    for metric in metrics:
        print(metric, ': ',  metrics[metric][0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set_dir', type=str, help='Path to the Speech Gold Alignment dataset folder.')
    parser.add_argument('--path_to_contribs', type=str, help='Path to a folder with a .npy file with the contributions for each sample in the dataset. Files names should be idx.npy.')
    parser.add_argument('--path_to_tokenized_targets', type=str, help='Path to a .txt folder with the tokenized target text used to teacher force the model. The sentence should be ordered by idx in the file.')
    parser.add_argument('--save_alignment_hyp', type=str, help='Path to a .txt file that will contain the hypotesis for the alignments.')
    parser.add_argument('--setting', type=str, help='Speech to Speech (s2s) or Speech to Text (s2t)')
    parser.add_argument('--translation_direction', type=str, help='en-de or de-en')
    args = parser.parse_args()
    
    assert args.translation_direction == 'en-de' or args.translation_direction == 'de-en', 'translation direction must be en-de or de-en'
    assert args.setting == 's2s' or args.setting == 's2t', 'The setting must be s2s or s2t.'
    if args.setting == 's2t':
        assert args.path_to_tokenized_targets, 'For s2t translation, it is required to include the argument path_to_tokenized_targets.'
    
    calculate_aercalculate_aer( args.test_set_dir, args.path_to_contribs, args.save_alignment_hyp, args.setting, args.translation_direction, args.path_to_tokenized_targets)
    

if __name__ == '__main__':
    main()