"""Prepare nextGEN pretrain dataset. 

Returns:
    random segments of melody and accompaniment.
"""
import os
import json
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import Counter


import sys
sys.path.append('..')
from utils.vocab import norm_pitch_vocab
from utils.tokenizer import BertTokenizer
from utils.event import Event, trunc_duration
from utils.common import get_file_list, load_note_event
from preprocess.segment import segment, get_sect_pickups


def flatten_measures(segment, bar_pad=True, max_measure_len=64):
    tokens = [segment['time_signature'], segment['tempo']]
    for i, measure in enumerate(segment['note']):
        tokens += measure
        if bar_pad:
            if i > 0 and i < len(segment['note']) - 1:
                n_pad = max_measure_len - len(measure)
                tokens += ['sep' for _ in range(n_pad)]
        tokens += ['bar']
    tokens[-1] = 'eos'
    return tokens


def verify_segment(segment, tokenizer, bar_pad=True, max_measure_len=64):
    is_valid = True
    for measure in segment['note']:
        if bar_pad and len(measure) > max_measure_len:
            is_valid = False
            break
        if tokenizer.has_irregular_token(measure):
            is_valid = False
            break
    return is_valid


def get_base_vocab(event_file_list):

    vocab_cnt = Counter()
    ts_set = set()
    tp_set = set()

    for event_file in tqdm(event_file_list, desc="Build Vocabulary"):
        event = load_note_event(event_file)

        # Update vocabulary count
        for i in event:
            vocab_cnt.update(event[i]['event'])
            ts_set.add(f"ts-{event[i]['time_signature']}")
            tp_set.add(f"tp-{event[i]['tempo']}")

    # Post process vocabulary
    pitch_vocab = [i for i in vocab_cnt if Event(i).event_type == "pitch"]
    vocab = set(norm_pitch_vocab(pitch_vocab, preset_pitch=False))

    vocab.update([i for i in vocab_cnt])
    vocab.update(ts_set)
    vocab.update(tp_set)
    base_vocab = sorted(list(vocab) + ["bar"])

    return base_vocab


def get_dataset(melody_file_list, tokenizer, pad_bar=True,
                max_measure_len=64, n_measure=8, n_hop=2, n_overlap=2):

    if not len(melody_file_list):
        return

    melody_dir = os.path.join(melody_file_list[0], "../..")
    melody_dir_base = os.path.basename(os.path.abspath(melody_dir))
    cpt_dir_base = "midi_no_repeat"

    paired_segments = []

    for melody_file in tqdm(melody_file_list):
        # Load pitch transposed, tempo normalized melody
        melody = load_note_event(melody_file)

        # Todo: reset duration token
        melody = trunc_duration(melody)

        # Load tempo/time signature change point file
        cpt_file = melody_file.replace(melody_dir_base, cpt_dir_base)
        with open(cpt_file) as f:
            cpts = json.load(f)['cpt']
        cpts.append({'measure': max(melody) + 1})
        pickups = get_sect_pickups(melody, cpts[:-1])

        for _ in range(0, n_measure, n_hop):

            segs_0 = segment(melody, cpts, pickups, n_measure, n_measure)

            next_cpts = deepcopy(cpts)
            next_cpts[0]['measure'] += n_measure - n_overlap
            if next_cpts[0]['measure'] >= next_cpts[1]['measure']:
                break
            segs_1 = segment(melody, next_cpts, pickups, n_measure, n_measure)

            cpts[0]['measure'] += n_hop

            n_pair = min(len(segs_0), len(segs_1))
            for i in range(n_pair):
                is_valid_0 = verify_segment(
                    segs_0[i], tokenizer, pad_bar, max_measure_len)
                is_valid_1 = verify_segment(
                    segs_1[i], tokenizer, pad_bar, max_measure_len)
                if is_valid_0 and is_valid_1:
                    seg_0 = flatten_measures(segs_0[i], pad_bar, max_measure_len)
                    seg_1 = flatten_measures(segs_1[i], pad_bar, max_measure_len)
                    paired_segments.append(
                        (tokenizer.convert_tokens_to_ids(seg_0),
                         tokenizer.convert_tokens_to_ids(['next'] + seg_1))
                    )

    return paired_segments


if __name__ == "__main__":

    DATA_DIR = "../../sonata-dataset-phrase"

    melody_dir = os.path.join(DATA_DIR, "norm_skyline_no_repeat")
    acc_dir = os.path.join(DATA_DIR, "norm_acc_no_repeat")

    melody_file_list = get_file_list(melody_dir)
    acc_file_list = get_file_list(acc_dir)

    # Get Cleaned Vocabulary
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab_melody_acc.txt")
    if not os.path.exists(base_vocab_file):
        base_vocab = get_base_vocab(melody_file_list + acc_file_list)
        base_vocab = ['next', 'key-mod', 'sect-end'] + base_vocab
        with open(base_vocab_file, "w") as f:
            for token in base_vocab:
                f.write(token + "\n")
    else:
        print(f"Load base vocabulary from {base_vocab_file}.")
        with open(base_vocab_file) as f:
            base_vocab = f.read().splitlines()

    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    max_seq_len = 512
    max_measure_len = 40
    n_measure = max_seq_len // max_measure_len
    n_hop = 2
    n_overlap = 2
    pad_bar = True

    # Get overlapped segments
    out_dir = os.path.join(DATA_DIR, "next_phrase_melody")
    dataset_dir = os.path.join(out_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    print("Get Overlapped Segments")
    print(f"# Measure per segments: {n_measure}")
    print(f"Hop size: {n_hop} (measures)")

    # Make dataset
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset_split.csv"))
    for split in ['train', 'val']:
        print(f"Process {split} set")
        melody_file_list = [os.path.join(melody_dir, fname)
                            for fname in df[df['split'] == split]['fname']]

        segment_pairs = get_dataset(melody_file_list, tokenizer, pad_bar,
                                    max_measure_len, n_measure, n_hop, n_overlap)

        with open(os.path.join(dataset_dir, f"{split}_{max_seq_len}_pad.json"), "w") as f:
            json.dump(segment_pairs, f)

    split = 'test'
    print(f"Process {split} set")
    melody_file_list = [os.path.join(melody_dir, fname)
                        for fname in df[df['split'] == split]['fname']]
    segment_pairs = get_dataset(melody_file_list, tokenizer, pad_bar,
                                max_measure_len, n_measure, n_measure, n_overlap)

    with open(os.path.join(dataset_dir, f"{split}_512_pad.json"), "w") as f:
        json.dump(segment_pairs, f)
