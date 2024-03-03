#!/usr/bin/env python3

# Code based on the original Jupyter Notebook developed by Aleix Sant

import os
import io
import gc
import re
import json
import random
import shutil
import numpy as np
from pathlib import Path
from itertools import groupby
from typing import List, Tuple
from sacremoses import MosesDetokenizer
from collections import Counter, defaultdict, deque

import torch
import torchaudio

from TTS.api import TTS


def get_durations(logw, x_mask, tts):
    w = torch.exp(logw) * x_mask * tts.synthesizer.tts_model.length_scale
    w_ceil = torch.ceil(w)
    return w_ceil


# Every '.' we encounter, VITS adds > 100 zeros.
# This function removes these zeros for posterior alignment with word_tuples
def postprocess_wav(wav_list: List[float], sentence: str) -> List[float]:
    wav = np.trim_zeros(np.array(wav_list))
    punctuation_marks = (
        Counter(sentence)["."]
        + Counter(sentence)["!"]
        + Counter(sentence)["?"]
        + Counter(sentence)["..."]
    )
    if punctuation_marks > 1:
        res = np.array([])
        for k, v in groupby(wav):
            v = list(v)
            if not (k == 0 and len(v) > 100):
                res = np.append(res, v)
        return res
    else:
        return wav


def sent_pho_dur_to_word_tuples(
    sent_pho_dur: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    word_duration = 0
    current_word = ""
    word_tuples = []
    for i in range(len(sent_pho_dur)):
        # Last element
        if i == len(sent_pho_dur) - 1:
            if sent_pho_dur[i][0] != "<BLNK>":
                current_word += sent_pho_dur[i][
                    0
                ]  # We omit <BLNK> when saving the word phonemes
            word_duration += sent_pho_dur[i][1]
            word_tuples.append((current_word, word_duration))
        # Other elements
        else:
            # Case: two subsentences (w/o word spacing due to concatenation) --> separate last word of subsent 1 from first word of subsent 2
            if sent_pho_dur[i][0] == "<BLNK>" and sent_pho_dur[i + 1][0] == "<BLNK>":
                word_duration += sent_pho_dur[i][1]
                word_tuples.append((current_word, word_duration))
                current_word = ""
                word_duration = 0
            # Case: no word spacing (all phonemes)
            # elif sent_pho_dur[i][0] not in [' ']:
            elif sent_pho_dur[i][0] != " ":
                word_duration += sent_pho_dur[i][1]
                if sent_pho_dur[i][0] != "<BLNK>":
                    current_word += sent_pho_dur[i][0]
            # Case: word spacing (' ')
            else:
                word_duration += sent_pho_dur[i][1] / 2
                word_tuples.append((current_word, word_duration))
                current_word = ""
                word_duration = sent_pho_dur[i][1] / 2
    return word_tuples  # (phon_word, dur_word)


def word_tuples_to_ini_end_clips(
    word_tuples: List[Tuple[str, int]], wav: List[float], sample_rate: float
) -> List[Tuple[float, float]]:
    duration_unit = len(wav) / sum([word_tuples[i][1] for i in range(len(word_tuples))])
    word_ini_end_clips = []
    start_idx = 0.0
    for word, dur_word in word_tuples:
        ini_sec = round(start_idx / sample_rate, 2)
        end_sec = round((start_idx + duration_unit * dur_word) / sample_rate, 2)
        word_ini_end_clips.append([ini_sec, end_sec])
        start_idx = start_idx + duration_unit * dur_word
    return word_ini_end_clips


def find_idxs_two_phon_words_merged(words: List[str], en_tts: TTS) -> List[int]:
    idx_list = []
    prev_phon_words = []
    # print("List of words:", words)
    for idx_word in range(len(words)):
        if idx_word != len(words) - 1:
            two_consec_words = (
                "".join(words[idx_word]) + " " + "".join(words[idx_word + 1])
            )
            tokenizer = en_tts.synthesizer.tts_model.tokenizer
            phon_words = tokenizer.decode(
                tokenizer.text_to_ids(two_consec_words, language="en")
            ).replace("<BLNK>", "")
            phon_words = phon_words.split(" ")
            # Get the first index of the two consec words that have been phonemized (merged) into 1 word,
            # given that the previous two consec words are correctly phonemized (not merged), and without considering
            # the cases of '-'. Hyphen ('-') always is removed in phonemization, and so, we always have 1 word after
            # phonemization. We don't save this latter case.
            if (
                len(phon_words) == 1
                and len(prev_phon_words) != 1
                and words[idx_word + 1] != "-"
            ):
                idx_list.append(idx_word)
                # print(f'The following words have been merged in phonemization: {two_consec_words}')
            prev_phon_words = phon_words  # save previous phonemization
    return idx_list  # Idxs of the first word of the merged phonemized words (idx ref wrt Moses detokenized sentence)


def find_idxs_three_phon_words_merged(words: List[str], en_tts: TTS) -> List[int]:
    idx_list = []
    prev_phon_words1 = []
    prev_phon_words2 = []
    # print("List of words:", words)
    for idx_word in range(len(words) - 2):
        if idx_word != len(words) - 2:
            three_consec_words = (
                "".join(words[idx_word])
                + " "
                + "".join(words[idx_word + 1])
                + " "
                + "".join(words[idx_word + 2])
            )
            tokenizer = en_tts.synthesizer.tts_model.tokenizer
            phon_words = tokenizer.decode(
                tokenizer.text_to_ids(three_consec_words, language="en")
            ).replace("<BLNK>", "")
            phon_words = phon_words.split(" ")
            if (
                len(phon_words) == 1
                and len(prev_phon_words1) != 1
                and len(prev_phon_words2) != 1
                and (words[idx_word + 1] != "-" or words[idx_word + 2] != "-")
            ):
                idx_list.append(idx_word)
                # print(f'The following words have been merged in phonemization: {three_consec_words}')
            prev_phon_words1 = phon_words  # save last phonemization
            prev_phon_words2 = prev_phon_words1  # save penultimate phonemization
    return idx_list  # Idxs of the firt word of the merged phonemized words (idx ref wrt Moses detokenized sentence)


def find_idxs_n_pos_multiple_phon_words(words: List[str], en_tts: TTS) -> List[int]:
    phon_percentage = "pɚsˈɛnt"  # % phonemization.
    idx_list = []
    n_pos_extra_list = []
    for idx_word in range(len(words)):
        tokenizer = en_tts.synthesizer.tts_model.tokenizer
        phon_word = tokenizer.decode(
            tokenizer.text_to_ids(words[idx_word], language="en")
        ).replace("<BLNK>", "")
        phon_word = phon_word.split(" ")
        if len(phon_word) != 1:
            # Case without %: numbers, hyphenated compound words... (e.g., 30, 34, sister-in-law...)
            if (
                phon_percentage not in phon_word[-1]
            ):  # Use 'in' (not ==) because phon_word[-1] can have a stacked 'punctuation mark'
                idx_list.append((idx_word, None))  # Boolean refers to the %
                n_pos_extra_list.append(len(phon_word) - 1)
            # Case with %: numbers followed by % (e.g., 30%, 24%...)
            else:
                idx_list.append((idx_word, "%"))  # Boolean refers to the %
                # Last phon word is "pɚsˈɛnt" (%). Don't want to merge it in the clip, so n_pos_extra is one unit less.
                # e.g., 32% (thirty two -percent-), 124% (one hundred twenty four -percent-)...
                n_pos_extra_list.append(len(phon_word) - 2)
            # print(f'The following word have been phonemized in multiple words (i.e., multiple clips): {words[idx_word]}')
    return (
        idx_list,
        n_pos_extra_list,
    )  # Idxs of the words phonemized in multiple phonemized words (idx ref wrt Moses detokenized sentence)


def merge_list_of_tuples(idx_copy_two, idx_copy_three):
    i, j = 0, 0
    merged_list = []
    while i < len(idx_copy_two) and j < len(idx_copy_three):
        if idx_copy_two[i][0] < idx_copy_three[j][0]:
            merged_list.append(idx_copy_two[i])
            i += 1
        else:  # idx_copy_three[j][0] < idx_copy_two[i][0]:
            merged_list.append(idx_copy_three[j])
            j += 1
    # Add remaining part of the non-empty list (either idx_copy_two or idx_copy_three)
    merged_list += [x for x in idx_copy_two[i:]] + [x for x in idx_copy_three[j:]]
    return merged_list


def reorganize_clips_after_phonemization(
    dataset_ini_end_clips, idx_list, n_pos_extra_list, idx_copy_two, idx_copy_three
) -> List[Tuple[float, float]]:
    # We reorganize the clips of the dataset_ini_end_clips taking into consideration the merge phonemizations (clip copies),
    # the multiple word phonemizations (considering having a % and without)

    # print(f'Idx_list: {idx_list}, N_pos_extra_lis: {n_pos_extra_list}')
    # print(f'Idx_copy_two: {idx_copy_two}, Idx_copy_three: {idx_copy_three}')

    idx_copy_two = [(idx, "2") for idx in idx_copy_two]
    idx_copy_three = [(idx, "3") for idx in idx_copy_three]

    idx_copy_list = merge_list_of_tuples(idx_copy_two, idx_copy_three)

    # Save idxs of multiple words phonemization with %
    idx_perc = [(idx, type_perc) for idx, type_perc in idx_list if type_perc == "%"]
    idx_copy_list_perc = merge_list_of_tuples(idx_copy_list, idx_perc)

    # Idx shifting to correctly copy clips when two/three phon words are merged.
    # Every time a copy idx is found, we need to sum 1/2 to the current and posterior idxs
    # due to the additional (copied) clip inserted in the list.
    # Moreover, every time we encoutnar a perc_idx, we need to sum 1 to skip the %.
    new_copy_list = []
    shift = 0
    cont_perc = 0
    for idx_copy, type_copy in idx_copy_list_perc:
        if type_copy == "2":
            new_copy_list.append((idx_copy - shift + cont_perc, "2"))
            shift += 1
        elif type_copy == "3":
            new_copy_list.append((idx_copy - shift + cont_perc, "3"))
            shift += 2
        else:  # type_copy == '%'
            cont_perc += 1

    if len(idx_list) > 0:
        # Reajustment of idxs due to the affectation the multiple phon words from a
        # single word
        new_idx_list = []
        shift = 0
        if len(idx_copy_list) > 0:
            for idx_joint, type_joint in idx_list:
                for idx_copy, type_copy in idx_copy_list:
                    if idx_joint > idx_copy:
                        if type_copy == "2":
                            shift += 1
                        else:  # == '3'
                            shift += 2
                new_idx_list.append((idx_joint - shift, type_joint))
                shift = 0
            idx_list = new_idx_list

        new_dataset = []  # equivalent to new_dataset_ini_end_clips
        idx_queue = deque(idx_list)
        n_extra_queue = deque(n_pos_extra_list)
        threshold = idx_queue[0][0]

        cont = 0
        cont_perc = 0
        for d in dataset_ini_end_clips:
            if (
                len(new_dataset) == 0
                or len(new_dataset) <= threshold
                or len(idx_queue) == 0
            ):
                new_dataset.append(d)
            else:
                if n_extra_queue[0] != 0:
                    new_dataset[-1][1] = d[1]
                    n_extra_queue[0] -= 1
                    cont += 1
                else:  # case single phon word of a number and %
                    new_dataset.append(d)
                if n_extra_queue[0] == 0:
                    if idx_queue[0][1] == "%":
                        cont_perc += 1
                    n_extra_queue.popleft()
                    idx_queue.popleft()
                    if len(idx_queue) != 0:
                        cont = 0
                        threshold = idx_queue[0][0] - cont + cont_perc
    else:
        new_dataset = dataset_ini_end_clips

    i = 0
    for idx, type_copy in new_copy_list:
        if type_copy == "2":
            new_dataset.insert(idx + i, new_dataset[idx + i])
            i += 1
        else:  # == '3'
            new_dataset.insert(idx + i, new_dataset[idx + i])
            new_dataset.insert(idx + i + 1, new_dataset[idx + i + 1])
            i += 2
    return new_dataset


def get_idx_punct_pos_s_token(tokens_sent):
    # We get the idx location in the Moses tokenized sentence of the punctuations
    # marks and the possessive 's. These tokens are treated differently as
    # punctuation marks are not assigned to any audio clip and words with
    # possessive 's are merged after Moses detokenization, and so, their phonemization
    idx_punc = []
    idx_pos_s = []
    punc_tok = [".", ",", "(", ")", '"', "'", "-", "?", "!", ":", ";"]
    for i, tok in enumerate(tokens_sent):
        if tok in punc_tok:
            idx_punc.append(i)
        if tok == "'s":
            idx_pos_s.append(i)
    return idx_punc, idx_pos_s


def add_none_clips(dataset_ini_end_clips, idx_tokens) -> List[Tuple[float, float]]:
    # Reajustment of idxs needed to align the Moses tokenized sentence to the phonemized word clips
    new_idx_tokens = []
    shift = 0
    for idx_tok in idx_tokens:
        new_idx_tokens.append(idx_tok - shift)
        shift += 1
    # Insertion of (None, None) tuples in the punctuation mark locations
    i = 0
    for idx in new_idx_tokens:
        dataset_ini_end_clips.insert(idx + i, (None, None))
        i += 1
    return dataset_ini_end_clips


def reorganize_clips_to_match_tokens(
    dataset_ini_end_clips, idx_punc, idx_pos_s
) -> List[Tuple[float, float]]:

    new_idx_pos_s = []
    shift = 0
    for idx_s in idx_pos_s:
        new_idx_pos_s.append(idx_s - shift)
        shift += 1

    if len(idx_punc) > 0:
        new_idx_punc = []
        shift = 0
        if len(idx_pos_s) > 0:
            for idx in idx_punc:
                for idx_s in idx_pos_s:
                    if idx > idx_s:
                        shift += 1
                new_idx_punc.append(idx - shift)
                shift = 0
            idx_punc = new_idx_punc
        new_dataset = add_none_clips(dataset_ini_end_clips, idx_punc)
    else:
        new_dataset = dataset_ini_end_clips

    # We add a copy of the clip in the "'s" token position because during the
    # phonemization the 's is merged with the previous word (e.g., Parliament's)
    i = 0
    for idx in new_idx_pos_s:
        new_dataset.insert(idx + i, new_dataset[idx + i - 1])  # copy previous clip
        i += 1
    return new_dataset


# JUST FOR VERIFICATION
# Function that modifies the length of the original words to match the length in
# the assert
def words_postprocessing(words: List[str]):
    counts = words.count("-")
    while counts > 0:
        words.remove("-")
        counts -= 1
    counts = sum(1 for word in words if "%" in word)
    while counts > 0:
        words.append("%")
        counts -= 1
    return words


def main():

    ROOT = Path("./dataset").absolute()
    NUM_SPEAKERS = 5
    SAMPLE_RATE = 16_000

    gpu = torch.cuda.is_available()

    en_tts_single = TTS("tts_models/en/ljspeech/vits", gpu=gpu)

    idx2char = en_tts_single.synthesizer.tts_model.tokenizer._characters._id_to_char
    sample_rate = en_tts_single.synthesizer.output_sample_rate

    def textenc_hook(module, input, output):
        textenc_in.append(input)
        textenc_out.append(output)

    def logw_hook(module, input, output):
        logw.append(output)

    en_tts_single.synthesizer.tts_model.text_encoder.register_forward_hook(textenc_hook)
    en_tts_single.synthesizer.tts_model.duration_predictor.register_forward_hook(
        logw_hook
    )

    en_tts_multi = TTS("tts_models/en/vctk/vits", gpu=gpu)

    idx2char = en_tts_multi.synthesizer.tts_model.tokenizer._characters._id_to_char
    sample_rate = en_tts_multi.synthesizer.output_sample_rate

    en_tts_multi.synthesizer.tts_model.text_encoder.register_forward_hook(textenc_hook)
    en_tts_multi.synthesizer.tts_model.duration_predictor.register_forward_hook(
        logw_hook
    )

    md = MosesDetokenizer(lang="en")

    with open(ROOT / "en", "r", encoding="iso-8859-1") as file:
        sentences = file.read().splitlines()

    # Exclude empty lines at the end
    while sentences and not sentences[-1]:
        sentences.pop()

    all_speakers = list(en_tts_multi.speakers)
    all_speakers[0] = "ED"

    sent_pho_dur = []
    dataset_ini_end_clips = []
    wavs = []
    for i, sent in enumerate(sentences):

        tokens_sent = sent.split()

        rec_sent = md.detokenize(tokens_sent)
        words = rec_sent.split()
        words_assert = words_postprocessing(words)

        sent_pho_dur = []
        dataset_ini_end_clips = []

        jsons_filepath = ROOT / "jsons"
        jsons_filepath.parent.mkdir(parents=True, exist_ok=True)

        # We create a dict variable (sent_dic) for every sentence and we save its content
        # in a json. Finally, we create a big dictionary unifying all of them.
        sent_dic = defaultdict(lambda: {})

        random.seed(i)
        speakers = random.sample(all_speakers, NUM_SPEAKERS)
        # speakers = [spk for spk in speakers if spk != 'ED\n' else 'ED'] no va aixo
        for j in range(NUM_SPEAKERS + 1):
            torch.manual_seed(10 * i + j)
            spk_name = f"vctk_spk_{speakers[j]}" if j < NUM_SPEAKERS else "lj_spk"
            wav_filepath = ROOT / "wavs" / f"{i:03d}" / f"{i:03d}_{spk_name}.wav"
            wav_filepath.parent.mkdir(parents=True, exist_ok=True)

            # textenc_in: n_sent , n_input (0: x), batch
            # textenc_out: n_sent , n_output (-1: x_mask), batch
            # logw: n_sent, batch
            textenc_in = []
            textenc_out = []
            logw = []

            if spk_name == "lj_spk":
                wav = en_tts_single.tts(text=rec_sent)
                old_sr = en_tts_single.synthesizer.output_sample_rate
            else:
                if spk_name == "vctk_spk_ED":
                    wav = en_tts_multi.tts(text=rec_sent, speaker="ED\n")
                    old_sr = en_tts_multi.synthesizer.output_sample_rate
                else:
                    wav = en_tts_multi.tts(text=rec_sent, speaker=speakers[j])
                    old_sr = en_tts_multi.synthesizer.output_sample_rate

            wav = postprocess_wav(wav, sent)

            final_wav = torchaudio.functional.resample(
                torch.Tensor(wav), old_sr, SAMPLE_RATE
            ).unsqueeze(0)
            torchaudio.save(wav_filepath, final_wav, sample_rate=SAMPLE_RATE)

            phonemes = []
            durations = []
            for inp, out, lw in zip(textenc_in, textenc_out, logw):
                phonemes.extend(inp[0].squeeze(0).tolist())
                subsent_durations = get_durations(lw, out[-1], en_tts_multi)[0][
                    0
                ].tolist()
                durations.extend(subsent_durations)

            sent_pho_dur.append([(idx2char[p], d) for p, d in zip(phonemes, durations)])

            word_tuples = sent_pho_dur_to_word_tuples(sent_pho_dur[j])
            dataset_ini_end_clips.append(
                word_tuples_to_ini_end_clips(word_tuples, wav, sample_rate)
            )

            idx_list, n_pos_extra_list = find_idxs_n_pos_multiple_phon_words(
                words, en_tts_multi
            )
            idx_copy_list_two = find_idxs_two_phon_words_merged(words, en_tts_multi)
            idx_copy_list_three = find_idxs_three_phon_words_merged(words, en_tts_multi)
            n_shifts = sum(n_pos_extra_list)
            assert len(words_assert) == (
                len(dataset_ini_end_clips[j])
                + len(idx_copy_list_two)
                + 2 * len(idx_copy_list_three)
                - n_shifts
            ), "Wrong idx detection of conflictive phonemization cases"

            dataset_ini_end_clips[j] = reorganize_clips_after_phonemization(
                dataset_ini_end_clips[j],
                idx_list,
                n_pos_extra_list,
                idx_copy_list_two,
                idx_copy_list_three,
            )
            assert len(words_assert) == len(
                dataset_ini_end_clips[j]
            ), "Wrong processing of clips after sentence phonemization"

            idx_punc, idx_pos_s = get_idx_punct_pos_s_token(tokens_sent)
            dataset_ini_end_clips[j] = reorganize_clips_to_match_tokens(
                dataset_ini_end_clips[j], idx_punc, idx_pos_s
            )

            # wavs.append(wav) --> uncomment in case to save examples and use the commented
            # loop below (last cell) to hear some audio clips after alignment
            sent_dic[f"{i:03d}"][spk_name] = [
                (word, cli[0], cli[1])
                for (word, cli) in zip(tokens_sent, dataset_ini_end_clips[j])
            ]

            gc.collect()
            if gpu:
                torch.cuda.empty_cache()

        if not os.path.exists(jsons_filepath):
            os.makedirs(jsons_filepath)

        with open(
            os.path.join(jsons_filepath, f"dataset_{i:03d}.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(sent_dic, file, indent=4, ensure_ascii=False)

        del sent_dic

    merged_dict = {}
    for json_f in jsons_filepath.iterdir():
        with open(json_f, encoding="UTF-8") as file:
            json_sent = json.load(file)
        merged_dict = {**merged_dict, **json_sent}

    dataset = merged_dict

    with open(ROOT / "en_dataset.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
    shutil.rmtree(jsons_filepath)

if __name__ == "__main__":
    main()
