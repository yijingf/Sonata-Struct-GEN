"""
nextGEN_pretrain_dataset.py

Description:
    Segments normalized melody event files (produced by normalize_notes.py) into *pairs of
    consecutive segments* for preparing a pretraining dataset for nextGEN. Score–MIDI mapping
    files are used to detect time‐signature and tempo change points that guide segmentation.

    Default Input:
        - Normalized melody event files:
            DATA_DIR/event_part/norm_melody/<composer>/
        - Score MIDI mapping files (used for tempo/time‐signature changes):
            DATA_DIR/midi/<composer>/
        - Normalized accompaniment event files (used to build vocabulary

    Output:
        - Segmented phrases saved to:
            DATA_DIR/dataset/nextGEN_pretrain/

Notes:
    It is assumed that time signature and tempo remain constant within a phrase.
    A new segment is started whenever a change in time signature or tempo is detected.

Usage:
    python3 nextGEN_pretrain_dataset.py
        [--measure_len 40]
        [--seq_len 512]
        [--n_hop 2]
        [--n_overlap 2]
        [--no_bar_pad]

Arguments:
    --measure_len    Maximum number of tokens per measure. Defaults to 40.
    --seq_len        Maximum number of tokens per sequence. Defaults to 512.
    --n_hop          Hop size between segments in bars. (default: 2)
    --n_overlap      # of overlapping bars between consecutive segments. (default: 2)
    --no_bar_pad     Disable bar-level padding. If not set, bar padding is enabled.


Example:
    python3 nextGEN_pretrain_dataset.py --measure_len 40 --seq_len 512 --n_hop 2 --n_overlap 2
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vocab import build_vocab_from_event
from utils.tokenizer import BertTokenizer
from utils.event import trunc_duration
from utils.common import get_file_list, load_note_event
from segment import segment, get_sect_pickups
from generate_data_split import generate_split

# Constants
from utils.constants import DATA_DIR


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


def get_melody_dataset(melody_dir, pad_bar=True, measure_len=64, n_measure=8, n_hop=2, n_overlap=2):

    file_list = get_file_list(melody_dir)
    if not len(file_list):
        return

    cpt_dir = os.path.basename(DATA_DIR, "midi")

    paired_segments = []

    for fname in tqdm(file_list):
        # Load pitch transposed, tempo normalized melody
        melody = load_note_event(fname)

        # Todo: reset duration token
        melody = trunc_duration(melody)

        # Load tempo/time signature change point file
        cpt_file = fname.replace(melody_dir, cpt_dir)
        with open(cpt_file) as f:
            cpts = json.load(f)['cpt']
        cpts.append({'measure': max(melody) + 1})
        pickups = get_sect_pickups(melody, cpts[:-1])

        for _ in range(0, n_measure, n_hop):

            # n_hop = n_measure in both seg_0 and seg_1 to make sure they matche
            segs_0 = segment(melody, cpts, pickups, n_measure, n_measure)

            next_cpts = deepcopy(cpts)
            next_cpts[0]['measure'] += n_measure - n_overlap
            if next_cpts[0]['measure'] >= next_cpts[1]['measure']:
                break
            segs_1 = segment(melody, next_cpts, pickups, n_measure, n_measure)

            cpts[0]['measure'] += n_hop

            n_pair = min(len(segs_0), len(segs_1))
            for i in range(n_pair):
                is_valid_0 = verify_segment(segs_0[i], tokenizer, pad_bar, measure_len)
                is_valid_1 = verify_segment(segs_1[i], tokenizer, pad_bar, measure_len)
                if is_valid_0 and is_valid_1:
                    seg_0 = flatten_measures(segs_0[i], pad_bar, measure_len)
                    seg_1 = flatten_measures(segs_1[i], pad_bar, measure_len)
                    paired_segments.append(
                        (tokenizer.convert_tokens_to_ids(seg_0),
                         tokenizer.convert_tokens_to_ids(['next'] + seg_1)))

    return paired_segments


def main(seq_len=512, measure_len=40, n_hop=2, n_overlap=2, bar_pad=True):
    # Load the train/val/test split
    data_split_fname = os.path.join(DATA_DIR, "dataset_split.csv")
    if not os.path.exists(data_split_fname):
        print("Train/val/test split file not found. Create one with `generate_split.py`.")
        return
    df = pd.read_csv(data_split_fname)

    # Constant
    n_measure = seq_len // measure_len

    melody_dir = os.path.join(DATA_DIR, "event_part", "norm_melody")
    acc_dir = os.path.join(DATA_DIR, "event_part", "norm_acc")
    melody_file_list = get_file_list(melody_dir)
    acc_file_list = get_file_list(acc_dir)

    # Vocabulary
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    if not os.path.exists(base_vocab_file):
        # Get Cleaned Vocabulary
        base_vocab = build_vocab_from_event(melody_file_list + acc_file_list)
        base_vocab = ['next', 'key-mod', 'sect-end'] + base_vocab
        with open(base_vocab_file, "w") as f:
            for token in base_vocab:
                f.write(token + "\n")
    else:
        print(f"Load base vocabulary from {base_vocab_file}.")
        with open(base_vocab_file) as f:
            base_vocab = f.read().splitlines()

    # Train Tokenizer
    global tokenizer
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    # Get overlapped segments and make datasets
    dataset_dir = os.path.join(DATA_DIR, "dataset", "nextGEN_pretrain")
    os.makedirs(dataset_dir, exist_ok=True)

    print("Get Overlapped Segments")
    print(f"# Measure per segments: {n_measure}")
    print(f"Hop size: {n_hop} in measures")

    for split in ['train', 'val']:
        print(f"Process {split} set")
        file_list = [os.path.join(melody_dir, fname) for fname in df[df['split'] == split]['fname']]
        pairs = get_melody_dataset(file_list, bar_pad, measure_len, n_measure, n_hop, n_overlap)

        with open(os.path.join(dataset_dir, f"{split}_{seq_len}_pad.json"), "w") as f:
            json.dump(pairs, f)

    print(f"Process {split} set")
    file_list = [os.path.join(melody_dir, fname) for fname in df[df['split'] == "test"]['fname']]
    pairs = get_melody_dataset(file_list, bar_pad, measure_len, n_measure, n_measure, n_overlap)

    with open(os.path.join(dataset_dir, f"{split}_512_pad.json"), "w") as f:
        json.dump(pairs, f)

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure_len", type=int, default=40,
                        help="Maximum number of tokens per measure. Defaults to 40")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Maximum number of tokens per sequence. Defaults to 512")
    parser.add_argument("--n_hop", type=int, default=2,
                        help="Hop size between segments in bars. Defaults to 2.")
    parser.add_argument("--n_overlap", type=int, default=2,
                        help="# of overlapping bars between consecutive segments. Defaults to 2.")
    parser.add_argument("--no_bar_pad", action="store_true",
                        help="Disable bar-level padding.")

    args = parser.parse_args()
    main(args.seq_len, args.measure_len, args.n_hop, args.n_overlap, not args.no_bar_pad)
