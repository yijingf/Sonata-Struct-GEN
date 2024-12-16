"""
Make dataset for MASS model.

Output:
train/validation dataset as JSON file in DATA_DIR/dataset with the following structure:
`[
    "token_ids": [],
    "center_mask_idx": [],
    "rand_mask_idx": []
], 
...
`
where `center_mask_idx` are indices of masked tokens in the continuous measures in the middle of a phrase, `rand_mask_idx` are indices of masked tokens in random measures. 

This script also generates a base vocabluary list in DATA_DIR/vocab/base_vocab.txt of normalized music event tokens. 

Usage:
# Generate dataset 80% of data as training set, sequence length of 512, bar-level padded.
python3 dataset.py [--split_ratio 0.8] [--seq_len 512] [--pad_bar]

"""

import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter

import sys
sys.path.append("..")
from utils.common import token2v
from utils.tokenizer import BertTokenizer
from utils.event import flatten_measures, map_measure_to_token

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


def make_dataset(tokenizer, df, data_dir, split='train', seq_len=256):
    dataset = []
    for _, row in df[df['split'] == split].iterrows():
        fname = os.path.join(data_dir, row['fname'])

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


def make_masked_dataset(tokenizer, df, data_dir, split,
                        max_measure_len=64, pad_bar=True,
                        seq_len=512, mask_ratio=0.75,):

    dataset = []

    for _, row in df[df['split'] == split].iterrows():
        fname = os.path.join(data_dir, row['fname'])

        with open(fname) as f:
            phrases = json.load(f)

        for phrase in phrases:

            if len(phrase['note']) < 2:
                continue

            tokens = flatten_measures(phrase,
                                      eos_token=tokenizer.eos_token,
                                      pad_bar=pad_bar,
                                      bar_pad_token=tokenizer.sep_token,
                                      max_measure_len=max_measure_len)

            if len(tokens) >= seq_len:  # roberta pos id starts from 1
                continue

            if tokenizer.has_irregular_token(tokens):
                continue

            # Get indices of first onset token in every measure
            idxs = np.append(2, np.where(np.array(tokens) == 'bar')[0] + 1)
            idxs = np.append(idxs, len(tokens))

            if len(phrase['note']) == 2:
                center_mask_idx = []
            else:
                center_mask_idx = mask_center_measure(idxs)

            rand_mask_idx = mask_rand_measure(idxs, mask_ratio)

            entry = {"token_ids": tokenizer.convert_tokens_to_ids(tokens),
                     "center_mask_idx": center_mask_idx,
                     "rand_mask_idx": rand_mask_idx}
            dataset.append(entry)

    return dataset


def split_dataset(data_dir, split_ratio=0.8):
    """Split dataset by file name.

    Args:
        data_dir (_type_): _description_
        split_ratio (float, optional): _description_. Defaults to 0.8.

    Returns:
        _type_: _description_
    """
    fname_list = []

    composers = os.listdir(data_dir)
    for composer in composers:
        fname_list += glob(os.path.join(data_dir, f'{composer}/*.json'))

    fname_list = sorted(fname_list)

    n_file = len(fname_list)

    idx = list(range(n_file))
    random.shuffle(idx)

    n_train = int(n_file * split_ratio)
    train_idx = idx[: n_train]
    val_idx = idx[n_train:]

    fname_list = ['/'.join(i.split("/")[-2:]) for i in fname_list]
    df = pd.DataFrame({'fname': fname_list})
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'

    return df


def get_vocab(seg_dir, min_freq=20):
    """_summary_

    Args:
        seg_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    composers = os.listdir(seg_dir)

    vocab_cnt = Counter()

    for composer in composers:

        seg_files = glob(os.path.join(seg_dir, composer, "*.json"))
        for seg_file in seg_files:

            with open(seg_file) as f:
                segments = json.load(f)

            for seg in segments:
                for measure in seg['note']:
                    vocab_cnt.update(measure)

                vocab_cnt.update([seg['time_signature']])
                vocab_cnt.update([str(seg['tempo'])])

    # Reject 1/11, 1/5, 1/7 notes or notes with occurences < 20
    tokens = list(vocab_cnt.keys())
    for token in tokens:
        if token[0] in ['o', 'd']:
            div = token2v(token).denominator

            if not div % 5 or not div % 11 or not div % 7:
                vocab_cnt.pop(token)
                continue

        if vocab_cnt[token] < min_freq:
            vocab_cnt.pop(token)

    vocab = sorted(list(vocab_cnt.keys()) + ['bar'])

    return vocab


def main(split_ratio, seq_len=512, pad_bar=False):

    seg_dir = os.path.join(DATA_DIR, "segment")

    # Split train/validation dataset.
    fname_data_split = os.path.join(DATA_DIR, "dataset_split.csv")
    if not os.path.exists(fname_data_split):
        df = split_dataset(seg_dir, split_ratio)
        df.to_csv(fname_data_split, index=False)
    else:
        df = pd.read_csv(fname_data_split)

    # Get base vocabulary from the normalized event
    vocab_fname = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    if os.path.exists(vocab_fname):
        with open(vocab_fname) as f:
            base_vocab = f.read().splitlines()

    else:
        base_vocab = get_vocab(seg_dir)
        # Save to file
        with open(vocab_fname, "w") as f:
            for i in base_vocab:
                f.write(i + "\n")

    # Train Tokenizer
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    # Make dataset
    out_dir = os.path.join(DATA_DIR, "dataset")
    splits = ['train', 'val']

    for split in splits:
        dataset = make_masked_dataset(tokenizer, df, seg_dir,
                                      split=split, seq_len=seq_len, pad_bar=pad_bar)
        if pad_bar:
            fname = f"{split}_{seq_len}_masked_pad.json"
        else:
            fname = f"{split}_{seq_len}_masked.json"

        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(dataset, f)
    # for split in splits:
    #     datasets[split] = make_dataset(tokenizer, df, seg_dir,
    #                                     split=split, seq_len=seq_len)
    #     fnames[split] = os.path.join(out_dir, f"{split}_{seq_len}.json")

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_ratio", dest="split_ratio", type=float, default=0.8,
        help="Train/validation split ratio. Default to 0.8.")
    parser.add_argument("--seq_len", dest="sequence_len", type=float,
                        default=512, help="Max sequence length. Default to 512.")
    parser.add_argument(
        "--no_pad_bar", dest="no_pad_bar", action="store_true",
        help="Without bar-level padding. Default to True.")

    args = parser.parse_args()
    main(args.split_ratio, args.seq_len, args.pad_bar)
