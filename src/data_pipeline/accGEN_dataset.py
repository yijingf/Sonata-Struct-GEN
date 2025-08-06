"""
accGEN_dataset.py

Description:
    Prepares a dataset for training the accGEN model using pairs of aligned melody and accompaniment segments.

Default Input:
    - Normalized melody event files:
        DATA_DIR/event_part/norm_melody/<composer>/
    - Normalized accompaniment event files:
        DATA_DIR/event_part/norm_acc/<composer>/
    - Score MIDI mapping files (used for tempo and time-signature changes):
        DATA_DIR/midi/<composer>/

Output:
    - Segmented phrases saved to:
        DATA_DIR/dataset/accGEN/

Usage:
    python3 accGEN_dataset.py
        [--measure_len 40]
        [--seq_len 512]
        [--seg_len None]
        [--n_hop 2]
        [--no_bar_pad]

Arguments:
    --measure_len    Maximum number of tokens per measure. Defaults to 40.
    --seq_len        Maximum number of tokens per sequence. Defaults to 512.
    --seg_len        Number of measures per segment. Defaults to None; internally set to seq_len // measure_len if not provided.
    --n_hop          Hop size between segments in bars. Defaults to 2.
    --no_bar_pad     Disable bar-level padding. If not set, padding is enabled.

Example:
    python3 accGEN_dataset.py --measure_len 40 --seq_len 512 --seg_len 8 --n_hop 2
"""

import os
import json
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.event import expand_score
from utils.common import load_event, load_note_event, normalize_tp, get_file_list
from utils.tokenizer import BertTokenizer

from dataset import format_segments
from segment import segment, get_sect_pickups


from utils.constants import DATA_DIR
event_dir = os.path.join(DATA_DIR, "event")
acc_dir = os.path.join(DATA_DIR, "event_part", "norm_acc")
melody_dir = os.path.join(DATA_DIR, "event_part", "norm_melody")
midi_dir = os.path.join(DATA_DIR, "midi")


def get_filtered_idx(segments, base_vocab, max_measure_len=64, max_len_seq=512):
    """Filter by segment length and vocabulary

    Args:
        segments (_type_): _description_
    """
    remove_idx = []

    for i, segment in enumerate(segments):

        len_seq = 0
        for j, measure in enumerate(segment['note']):

            if j == 0 or j == len(segment['note']) - 1:
                len_seq += len(measure)
            else:
                len_seq += max_measure_len

            if any([token not in base_vocab for token in measure]):
                remove_idx.append(i)
                break
            if len(measure) > max_measure_len:
                remove_idx.append(i)
                break

        # n - 1 bar token, eos token, ts/tp token
        if len_seq + len(segment['note']) + 2 >= max_len_seq:
            remove_idx.append(i)

    idx = set(range(len(segments))).difference(remove_idx)
    return sorted(list(idx))


def main(seq_len=512, measure_len=40, n_hop=2, seg_len=None, pad_bar=True):

    seg_len = seg_len or (seq_len // measure_len)

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
    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()

    # Tokenizer
    global tokenizer
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    output_dir = os.path.join(DATA_DIR, "dataset", "accGEN")
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'val']
    for split in splits:
        print(f"Process {split} set")
        melody_dataset, acc_dataset = [], []
        event_file_list = [os.path.join(event_dir, fname)
                           for fname in df[df['split'] == split]['fname']]

        for event_file in event_file_list:

            # Load full-score for getting section pickups
            score_event, mark = load_event(event_file)
            event, _ = expand_score(score_event, mark, "no_repeat")
            for i in event:
                event[i]['tempo'] = normalize_tp(event[i]['tempo'])

            # Load tempo/time signature change point file
            cpt_file = melody_file.replace(melody_dir, midi_dir)
            with open(cpt_file) as f:
                cpts = json.load(f)['cpt']
            cpts.append({'measure': max(melody) + 1})
            pickups = get_sect_pickups(melody, cpts[:-1])

            # Load expanded parts
            melody_file = event_file.replace(event_dir, melody_dir)
            melody = load_note_event(melody_file)
            acc_file = event_file.replace(event_dir, acc_dir)
            acc = load_note_event(acc_file)

            # Get segment pairs
            melody_segments = segment(melody, cpts, pickups, seg_len, n_hop)
            melody_idx = get_filtered_idx(melody_segments, base_vocab, measure_len)

            acc_segments = segment(acc, cpts, pickups, seg_len, n_hop)
            acc_idx = get_filtered_idx(acc_segments, base_vocab, measure_len)

            idx = list(set(acc_idx).intersection(melody_idx))

            filtered_melody_segments = [melody_segments[i] for i in idx]
            melody_dataset += format_segments(filtered_melody_segments,
                                              measure_len, seq_len, pad_bar, mask=False)

            filtered_acc_segments = [acc_segments[i] for i in idx]
            acc_dataset += format_segments(filtered_acc_segments,
                                           measure_len, seq_len, pad_bar, mask=False)

        melody_dataset_file = os.path.join(output_dir, f"melody_{split}_512_masked_pad.json")
        with open(melody_dataset_file, "w") as f:
            json.dump(melody_dataset, f)

        acc_dataset_file = os.path.join(output_dir, f"acc_{split}_512_masked_pad.json")
        with open(acc_dataset_file, "w") as f:
            json.dump(acc_dataset, f)

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure_len", type=int, default=40,
                        help="Maximum number of tokens per measure. Defaults to 40")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Maximum number of tokens per sequence. Defaults to 512")
    parser.add_argument("--seg_len", type=int, default=None,
                        help="# of measures per seg. Defaults to seq_len//measure_len.")
    parser.add_argument("--n_hop", type=int, default=2,
                        help="Hop size between segments in bars. Defaults to 2.")
    parser.add_argument("--no_bar_pad", action="store_true",
                        help="Disable bar-level padding.")

    args = parser.parse_args()
    main(args.seq_len, args.measure_len, args.n_hop, args.seg_len, not args.no_bar_pad)
