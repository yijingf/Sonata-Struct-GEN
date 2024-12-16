"""
Prepare accompaniment dataset
1. Segment full-score/melody into 8-bar segments with hop size of 2 bars. 
2. Get vocabulary
3. Make pairs of aligned quantized melody/skyline and full-score segment pairs
4. Build dataset
"""
import os
import json
import pandas as pd
from glob import glob
from collections import Counter

import sys
sys.path.append("..")
from utils.event import Event, expand_score
from utils.common import load_event, load_note_event, normalize_event
from utils.tokenizer import BertTokenizer
from kern_preprocess.segment import segment, get_sect_pickups
from kern_preprocess.dataset import make_masked_dataset


def clean_vocab(vocab_cnt, freq_thresh=50):
    """Remove irregular note onset/duration

    Args:
        vocab_cnt (collections.Counter): _description_
    """
    orig_vocab = list(vocab_cnt.keys())
    for token in orig_vocab:
        e = Event(token)
        if e.event_type != "pitch":
            if e.val == 0:
                continue
            if not all([e.val.denominator % 5,
                        e.val.denominator % 7,
                        e.val.denominator % 11,
                        e.val.denominator % 24,
                        e.val.denominator % 32]):
                del vocab_cnt[e.token_str]

    vocab = [i for i, v in vocab_cnt.items() if v >= freq_thresh]
    return vocab


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


if __name__ == "__main__":

    vocab_cnt = Counter()
    ts_set = set()
    tp_set = set()

    data_dir = "../../sonata-dataset-phrase"

    event_dir = os.path.join(data_dir, "event")
    melody_dir = os.path.join(data_dir, "quantized_skyline_no_repeat")

    out_dir = os.path.join(data_dir, "accompaniment")
    os.makedirs(out_dir, exist_ok=True)

    composers = ['mozart', 'haydn', 'beethoven', 'scarlatti']

    fullscore_seg_dir = os.path.join(out_dir, "fullscore_segments")
    melody_seg_dir = os.path.join(out_dir, "melody_segments")

    for composer in composers:

        os.makedirs(os.path.join(fullscore_seg_dir, composer), exist_ok=True)
        os.makedirs(os.path.join(melody_seg_dir, composer), exist_ok=True)

        for event_file in sorted(glob(os.path.join(event_dir, composer, "*.json"))):
            base_name = os.path.basename(event_file)

            # Load full-score
            score_event, struct = load_event(event_file)
            event, _ = expand_score(score_event, struct, "no_repeat")
            event = normalize_event(event, adjust_ts_tp=False)

            # Load tempo/time signature change point file
            cpt_file = event_file.replace("event", "midi_no_repeat")
            with open(cpt_file) as f:
                cpts = json.load(f)['onset']
            cpts.append({'measure': max(event) + 1})

            # Segment Full-score
            segments = segment(event, cpts)
            with open(os.path.join(fullscore_seg_dir, composer, base_name), "w") as f:
                json.dump(segments, f)

            # Segment melody
            melody = load_note_event(os.path.join(melody_dir, composer, base_name))
            melody = normalize_event(melody, adjust_ts_tp=False)
            pickups = get_sect_pickups(event, cpts[:-1])
            melody_segments = segment(melody, cpts, pickups)

            with open(os.path.join(melody_seg_dir, composer, base_name), "w") as f:
                json.dump(melody_segments, f)

            # Sanity check
            assert len(melody_segments) == len(segments), event_file

            # Update vocabulary count
            for i, phrase in enumerate(segments):
                for measure in phrase['note']:
                    vocab_cnt.update(measure)
                for measure in melody_segments[i]['note']:
                    vocab_cnt.update(measure)
                ts_set.add(phrase['time_signature'])
                tp_set.add(phrase['tempo'])

    vocab = clean_vocab(vocab_cnt, freq_thresh=50)
    base_vocab = sorted(vocab + list(tp_set) + list(ts_set) + ["bar"])

    with open(os.path.join(data_dir, "vocab", "base_vocab_accompaniment.txt")) as f:
        for token in base_vocab:
            f.write(token + "\n")

    # Filter Segments
    measure_len = 64

    filtered_fullscore_seg_dir = os.path.join(out_dir, "filtered_fullscore_segments")
    filtered_melody_seg_dir = os.path.join(out_dir, "filtered_melody_segments")

    for composer in composers:

        os.makedirs(os.path.join(filtered_fullscore_seg_dir, composer),
                    exist_ok=True)
        os.makedirs(os.path.join(filtered_melody_seg_dir, composer),
                    exist_ok=True)

        for seg_file in glob(os.path.join(fullscore_seg_dir, composer, "*.json")):

            # Full-score segments
            with open(seg_file) as f:
                segments = json.load(f)

            # Melody segments
            base_name = os.path.basename(seg_file)
            melody_seg_file = os.path.join(melody_seg_dir, composer, base_name)
            with open(melody_seg_file) as f:
                melody_segments = json.load(f)

            fullscore_idx = get_filtered_idx(segments, base_vocab, measure_len)
            melody_idx = get_filtered_idx(melody_segments, base_vocab, measure_len)
            idx = list(set(fullscore_idx).intersection(melody_idx))

            filtered_segments = [segments[i] for i in idx]
            filtered_melody_segments = [melody_segments[i] for i in idx]

            with open(os.path.join(filtered_fullscore_seg_dir, composer, base_name), "w") as f:
                json.dump(filtered_segments, f)

            with open(os.path.join(filtered_melody_seg_dir, composer, base_name), "w") as f:
                json.dump(filtered_melody_segments, f)

    # Make dataset

    df = pd.read_csv(os.path.join(data_dir, "dataset_split.csv"))

    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    dataset_dir = os.path.join(out_dir, "dataset")
    splits = ['train', 'val']

    for split in splits:
        dataset = make_masked_dataset(tokenizer, df, filtered_fullscore_seg_dir,
                                      split=split, seq_len=512, pad_bar=True)
        with open(os.path.join(dataset_dir, f"{split}_512_masked_pad.json"), "w") as f:
            json.dump(dataset, f)

    for split in splits:
        dataset = make_masked_dataset(tokenizer, df, filtered_melody_seg_dir,
                                      split=split, seq_len=512, pad_bar=True)
        with open(os.path.join(dataset_dir, f"melody_{split}_512_masked_pad.json"), "w") as f:
            json.dump(dataset, f)
