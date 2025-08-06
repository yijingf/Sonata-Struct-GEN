"""
dataset.py

Description:
    Creates a dataset for pretraining MASS-based model from normalized and segmented music events. 

    Each output JSON file contains training or validation samples with the following structure:
        [
            {
                "token_ids": [...],
                "center_mask_idx": [...],
                "rand_mask_idx": [...]
            },
            ...
        ]

    - `token_ids`: List of token IDs in a phrase.
    - `center_mask_idx`: Indices of masked tokens in a central, continuous masked region (center-masked).
    - `rand_mask_idx`: Indices of tokens randomly selected for masking.

    This script also generates a base vocabulary list at:
        DATA_DIR/vocab/base_vocab.txt
    containing all unique normalized music event tokens.

Usage:
    python3 dataset.py 
        --input_dir path/to/segments 
        --output_dir path/to/save/dataset 
        [--measure_len 48] 
        [--seq_len 512] 
        [--train_ratio 0.8] 
        [--no_bar_pad] 
        [--mask] 
        [--mask_ratio 0.75]

Arguments:
    --input_dir      Path to normalized input segments. (required)
    --output_dir     Directory to save the output dataset. (required)
    --measure_len    Maximum number of tokens per measure. Defaults to 48.
    --seq_len        Maximum number of tokens per sequence. Defaults to 512.
    --train_ratio    Proportion of data used for training. The rest is used for validation. Defaults to 0.8.
    --no_bar_pad     Disable bar-level padding. If not set, bar padding is enabled.
    --mask           Apply masking to tokens (center-masked and random-masked).
    --mask_ratio     Ratio of masking to apply, used when --mask is enabled. Defaults to 0.75.

Example:
    # Generate dataset with 80% training split, bar-padded, and sequence length 512
    python3 dataset.py --input_dir ./segments --output_dir ./dataset/mass_pretrain --measure_len 64

    # Generate dataset with masking and no bar padding
    python3 dataset.py --input_dir ./segments --output_dir ./dataset/mass_pretrain --no_bar_pad
"""
import os
import json
import random
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import get_file_list
from utils.tokenizer import BertTokenizer
from utils.event import flatten_measures, map_measure_to_token
from utils.vocab import build_vocab_from_segment
from generate_data_split import generate_split

# Constants
from utils.constants import DATA_DIR


def validate_phrase(phrase):
    i = 0
    while i < len(phrase):
        if phrase[i][0] == 'o':
            break
        else:
            i += 1

    j = len(phrase)
    while j > 0:
        if phrase[j - 1][0] == 'd':
            break
        elif phrase[j - 1] == 'bar':
            break
        elif phrase[j - 1] == 'eos':
            break
        else:
            j -= 1
    return phrase[i:j]


def split_phrase(phrase, seq_len=256):
    ts_tp, tokens = phrase[:2], phrase[2:]
    bar_pos = np.where(np.array(tokens) == 'bar')[0]

    phrases = []
    st = 0
    for i in range(1, len(bar_pos)):
        if bar_pos[i] - st + 3 > seq_len:
            segment = validate_phrase(tokens[st: bar_pos[i - 1] + 1])
            if len(segment) > seq_len - 1:
                segment = validate_phrase(segment[-seq_len + 2:])

            phrases.append(ts_tp + segment)
            st = bar_pos[max(0, i - 2)]

    if len(tokens) - st + 2 > seq_len:
        segment = ts_tp + validate_phrase(tokens[-seq_len + 3:])
    else:
        segment = ts_tp + validate_phrase(tokens[st:])

    phrases.append(segment)
    return phrases


def make_dataset(fname_list, seq_len=256):
    dataset = []
    for fname in fname_list:

        with open(fname) as f:
            phrases = json.load(f)

        for phrase in phrases['token']:
            if len(phrase) < 3:
                continue
            phrase += [tokenizer.eos_token]
            if len(phrase) <= seq_len:
                dataset.append(phrase)
            else:
                splitted_phrases = split_phrase(phrase, seq_len=seq_len)
                dataset += splitted_phrases
    return dataset


def mask_rand_measure(bar_idx, mask_ratio=0.5):
    n_measure = len(bar_idx) - 1

    # Randomly mask out measures
    n_masked_meaure = int(mask_ratio * n_measure)
    mask_measure = random.sample(range(n_measure), n_masked_meaure)
    mask_measure = np.array(sorted(mask_measure))

    mask_idx = map_measure_to_token(bar_idx, mask_measure)
    return mask_idx


def mask_center_measure(bar_idx):
    n_measure = len(bar_idx) - 1

    # Mask measures in the middle
    if n_measure < 3:
        return []

    # Mask around 50% bars in the middle of a sequence
    n_mask = int(n_measure / 2)
    mask_st_i = round(n_measure / 2 / 2)
    mask_measure = np.arange(mask_st_i, mask_st_i + n_mask)
    mask_idx = map_measure_to_token(bar_idx, mask_measure)

    return mask_idx


def make_masked_dataset(fname_list, measure_len=64, seq_len=512, pad_bar=True,
                        mask=True, mask_ratio=0.75):

    dataset = []

    for fname in fname_list:

        with open(fname) as f:
            segments = json.load(f)
        dataset += format_segments(segments, measure_len, seq_len, pad_bar, mask, mask_ratio)

    return dataset


def format_segments(segments, measure_len, seq_len, pad_bar, mask=True, mask_ratio=0.75):
    dataset = []
    for segment in segments:

        if len(segment['note']) < 2:
            continue

        tokens = flatten_measures(segment,
                                  eos_token=tokenizer.eos_token,
                                  pad_bar=pad_bar,
                                  bar_pad_token=tokenizer.sep_token,
                                  measure_len=measure_len)

        if len(tokens) >= seq_len:  # roberta pos id starts from 1
            continue

        if tokenizer.has_irregular_token(tokens):
            continue

        entry = {"token_ids": tokenizer.convert_tokens_to_ids(tokens)}
        if mask:
            # Get indices of first onset token in every measure
            idxs = np.append(2, np.where(np.array(tokens) == 'bar')[0] + 1)
            idxs = np.append(idxs, len(tokens))

            if len(segment['note']) == 2:
                center_mask_idx = []
            else:
                center_mask_idx = mask_center_measure(idxs)

            rand_mask_idx = mask_rand_measure(idxs, mask_ratio)

            entry["center_mask_idx"] = center_mask_idx
            entry["rand_mask_idx"] = rand_mask_idx

        dataset.append(entry)

    return dataset


def main(input_dir, output_dir, train_ratio=.8,
         measure_len=64, seq_len=512, bar_pad=True, mask=True, mask_ratio=0.75):

    # Load vocabulary
    vocab_fname = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    if os.path.exists(vocab_fname):
        with open(vocab_fname) as f:
            base_vocab = f.read().splitlines()
    else:
        # Get base vocabulary from event files
        base_vocab = build_vocab_from_segment(get_file_list(input_dir))
        with open(vocab_fname, "w") as f:
            for i in base_vocab:
                f.write(i + "\n")

    # Train Tokenizer
    global tokenizer
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    # Load the train/val/test split
    data_split_fname = os.path.join(DATA_DIR, "dataset_split.csv")
    if not os.path.exists(data_split_fname):
        # Create a sllit
        df = generate_split(get_file_list(input_dir), train_ratio)
        df.to_csv(data_split_fname, index=False)
    else:
        df = pd.read_csv(data_split_fname)

    # Make dataset
    splits = ['train', 'val']

    for split in splits:
        fname_list = [os.path.join(input_dir, row['fname'])
                      for _, row in df[df['split'] == split].iterrows()]

        dataset = make_masked_dataset(fname_list, measure_len, seq_len, bar_pad, mask, mask_ratio)

        masked_str = "_masked" if mask else ""
        pad_str = "_pad" if bar_pad else ""
        fname = f"{split}_{seq_len}{masked_str}{pad_str}.json"
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(dataset, f)

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to normalized input segments.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store the dataset")
    parser.add_argument("--measure_len", type=int, default=48,
                        help="Maximum number of tokens per measure. Defaults to 48")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Maximum number of tokens per sequence. Defaults to 512")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train set ratio. Defaults to 0.8.")
    parser.add_argument("--no_bar_pad", action="store_true",
                        help="Disable bar-level padding.")
    parser.add_argument("--mask", action="store_true",
                        help="Apply masking to segments.")
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                        help="Ratio of masking to apply. Defaults to 0.75.")

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        measure_len=args.measure_len,
        seq_len=args.seq_len,
        bar_pad=not args.no_bar_pad,
        mask=args.mask,
        mask_ratio=args.mask_ratio
    )
