"""
nextGEN_finetune_dataset.py

Description:
    Prepares a fine-tuning dataset for the nextGEN model using structurally segmented phrases.
    Unlike pretraining, this process leverages structural segmentation to extract musically 
    meaningful segments instead of using a fixed-length sliding window.


    Default Input:
        - Normalized melody event files:
            DATA_DIR/event_part/norm_melody/<composer>/
        - Score MIDI mapping files (used for tempo/time‐signature changes):
            DATA_DIR/midi/<composer>/
        - Structural segmentation files produced by:
            ./structural_segmentation/structural_segment.py
            DATA_DIR/struct_mark/

    Output:
        - Segmented phrases saved to:
            DATA_DIR/dataset/nextGEN_finetune/

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
    --no_bar_pad     Disable bar-level padding. If not set, bar padding is enabled.

Example:
    python3 nextGEN_finetune_dataset.py --measure_len 40 --seq_len 512 --n_hop 2 --n_overlap 2
"""

import os
import json
import numpy as np
import pandas as pd
from fractions import Fraction

import sys
sys.path.append("..")
from utils.common import load_note_event
from utils.tokenizer import BertTokenizer
from utils.align import get_t_bar, get_t_fin
from utils.event import trim_event, trunc_duration

# Constants
from utils.constants import DATA_DIR
melody_dir = os.path.join(DATA_DIR, "event_part", "norm_melody")
struct_dir = os.path.join(DATA_DIR, "struct_mark")
midi_dir = os.path.join(DATA_DIR, "midi")


def merge_bound(t_diff, t_min, t_max):
    new_t_diff = []
    res_t = 0
    for i, t in enumerate(t_diff):
        if t + res_t > t_min:
            new_t_diff += [t + res_t]
            res_t = 0
        else:
            res_t += t
    if res_t != 0:
        new_t_diff += [res_t]

    if len(new_t_diff) < 2:
        return new_t_diff

    if new_t_diff[-1] <= t_min and new_t_diff[-2] < t_max:
        new_t_diff[-2] = new_t_diff[-2] + new_t_diff[-1]
        return new_t_diff[:-1]
    return new_t_diff


def get_t_bounds(pred_bound, ts_cpt, n_min=2, n_max=10):
    for k, (bounds, _) in enumerate(pred_bound):

        lt_max_bar = []
        t_bounds = [0]
        for i, entry in enumerate(ts_cpt[:-1]):
            t_st, t_ed = entry['t'], ts_cpt[i + 1]['t']
            t_min = get_t_bar(**entry) * n_min
            t_max = get_t_bar(**entry) * n_max
            sect_bounds = bounds[np.logical_and(
                bounds[:, 0] >= t_st, bounds[:, 1] <= t_ed)]

            if len(sect_bounds):
                pre_bound = [[t_st, sect_bounds[0, 0]]]
                post_bound = [[sect_bounds[-1, -1], t_ed]]
                sect_bounds = np.concatenate([pre_bound, sect_bounds, post_bound])
            else:
                sect_bounds = [[t_st, t_ed]]

            t_diff = merge_bound(
                np.diff(sect_bounds, axis=1).reshape(-1),
                t_min, t_max)
            lt_max_bar += [t_max >= np.max(t_diff)]
            sect_bounds = np.cumsum([sect_bounds[0][0]] + t_diff)
            t_bounds += sect_bounds[1:].tolist()

        if all(lt_max_bar):
            break
    return t_bounds


def verify_ts_tp(seq):
    ts_tp = []
    for i in seq:
        ts = seq[i]['time_signature']
        tp = seq[i]['tempo']
        ts_tp += [f"{ts}-{tp}"]

    if len(set(ts_tp)) == 1:
        return True
    return False


def verify_phrase(phrase, tokenizer, seq_len=512):

    if tokenizer.has_irregular_token(phrase):
        return False

    if len(phrase) > seq_len:
        return False

    return True


def flatten_phrase(phrase, pad_bar=True, measure_len=40):
    i_st, i_ed = min(phrase), max(phrase)
    ts = f"ts-{phrase[i_st]['time_signature']}"
    tp = f"tp-{phrase[i_st]['tempo']}"
    tokens = [ts, tp]
    for i, measure in phrase.items():
        tokens += measure['event']
        if pad_bar:
            if i > i_st and i < i_ed:
                n_pad = measure_len - len(measure['event'])
                tokens += ['sep' for _ in range(n_pad)]
        tokens += ['bar']
    tokens[-1] = 'eos'
    return tokens


def get_phrase_pairs(event, struct, ts_cpt, pad_bar=True, seq_len=512, measure_len=40, n_hop=2):
    measure_cpt = [entry['measure'] for entry in ts_cpt]
    paired_phrases = []
    sect_names = list(struct.keys())
    for i_sect, sect in enumerate(sect_names):
        if i_sect == 0:
            st_1 = (0, 0)
        else:
            entries = struct[sect_names[i_sect - 1]]
            if not len(entries):
                e = struct[sect_names[i_sect - 2]][-1]
            else:
                e = entries[-1]
            st_1 = (e['end pos'][0], Fraction(e['end pos'][1]))
        for i, e in enumerate(struct[sect][:-1]):
            st_0 = st_1
            st_1 = (e['end pos'][0], Fraction(e['end pos'][1]))
            ed_1 = (struct[sect][i + 1]['end pos'][0],
                    Fraction(struct[sect][i + 1]['end pos'][1]))

            if st_1[0] in measure_cpt:
                # Cannot take 2 overlap primer
                continue

            # Sanity check: in extreme case scarlatti/L334K122
            seq_0 = trim_event(event, st_0, st_1)
            if not verify_ts_tp(seq_0):
                continue

            # seq_1 has two primer measrue overlapped with seq_0
            seq_1 = trim_event(event, (max(st_1[0] - n_hop, 0), st_1[1]), ed_1)
            if not verify_ts_tp(seq_1):
                continue

            phrase_0 = flatten_phrase(seq_0, pad_bar, measure_len)
            is_valid_0 = verify_phrase(phrase_0, tokenizer, seq_len)
            if not is_valid_0:
                continue

            phrase_1 = flatten_phrase(seq_1, pad_bar, measure_len)
            is_valid_1 = verify_phrase(phrase_1, tokenizer, seq_len)
            if not is_valid_1:
                continue

            if i == len(struct[sect]) - 2:
                type_token = "sect-end"
            elif struct[sect][i + 1]['key modulate']:
                type_token = "key-mod"
            else:
                type_token = "next"

            paired_phrases.append(
                (tokenizer.convert_tokens_to_ids(phrase_0),
                 tokenizer.convert_tokens_to_ids([type_token] + phrase_1))
            )

    return paired_phrases


def get_dataset(melody_file_list, pad_bar=True, seq_len=512, measure_len=40, n_hop=2):
    """Get pairs of phrases. The second phrase in a pair has n_hop of measures overlapped with the first phrase. 

    Args:
        melody_file_list (_type_): _description_
        pad_bar (bool, optional): _description_. Defaults to True.
        seq_len (int, optional): _description_. Defaults to 512.
        measure_len (int, optional): _description_. Defaults to 40.
        n_hop (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    phrase_pairs = []
    for event_file in melody_file_list:

        composer, basename = event_file.split(".json")[0].split("/")[-2:]
        event = load_note_event(event_file)
        event = trunc_duration(event)

        # Load score measure to midi measure mapping
        map_file = os.path.join(midi_dir, composer, f"{basename}.json")
        with open(map_file) as f:
            meta = json.load(f)

        ts_cpt = meta['cpt']
        idx_mapping = {int(i): v for i, v in meta['idx_mapping'].items()}

        # Renew end time
        t_fin = get_t_fin(max(idx_mapping), ts_cpt)
        ts_cpt.append({"measure": max(idx_mapping) + 1, 't': float(t_fin),
                       "tempo": ts_cpt[-1]["tempo"],
                       "time_signature": ts_cpt[-1]["time_signature"]})

        # Load predicted boundaries
        with open(os.path.join(struct_dir, f"{composer}-{basename}.json")) as f:
            struct = json.load(f)

        # Get merged boundaries
        phrase_pairs += get_phrase_pairs(event, struct, ts_cpt,
                                         pad_bar, seq_len, measure_len, n_hop)

    return phrase_pairs


def main(seq_len=512, measure_len=40, n_hop=2, pad_bar=True):

    melody_dir = os.path.join(DATA_DIR, "event_part", "norm_melody")

    # Sanity Check
    # Load the train/val/test split
    data_split_fname = os.path.join(DATA_DIR, "dataset_split.csv")
    if not os.path.exists(data_split_fname):
        print("Train/val/test split file not found. Create one with `generate_split.py`.")
        return
    df = pd.read_csv(data_split_fname)

    # Vocabulary
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    if not os.path.exists(base_vocab_file):
        print(f"{base_vocab_file} not found. Exit.")
        return

    # Tokenizer
    global tokenizer
    tokenizer = BertTokenizer()
    tokenizer.load_base_vocab(base_vocab_file)

    output_dir = os.path.join(DATA_DIR, "dataset", "nextGEN_finetune")
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val']:
        print(f"Process {split} set")
        melody_file_list = [os.path.join(melody_dir, fname)
                            for fname in df[df['split'] == split]['fname']]

        phrase_pairs = get_dataset(melody_file_list, pad_bar, seq_len, measure_len, n_hop)
        with open(os.path.join(output_dir, f"{split}_{seq_len}_pad.json"), "w") as f:
            json.dump(phrase_pairs, f)

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
    parser.add_argument("--no_bar_pad", action="store_true",
                        help="Disable bar-level padding.")

    args = parser.parse_args()
    main(args.measure_len, args.seq_len, args.n_hop, args.n_overlap, not args.no_bar_pad)
