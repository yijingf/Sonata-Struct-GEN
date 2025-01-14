import os
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter

import sys
sys.path.append('..')
from utils.vocab import clean_vocab
from utils.event import expand_score
from utils.common import get_file_list, load_event, load_note_event, normalize_event, normalize_tp
from utils.tokenizer import BertTokenizer
from preprocess.segment import segment, get_sect_pickups
from preprocess.melody_extract import skyline_variation

from utils.constants import COMPOSERS


def flatten_measures(segment, max_measure_len=64):
    tokens = [segment['time_signature'], segment['tempo']]
    for measure in segment['note']:
        tokens += measure
        n_pad = max_measure_len - len(measure)
        tokens += ['sep' for _ in range(n_pad)]
        tokens += ['bar']
    tokens[-1] = 'eos'
    return tokens


def verify_segment(segment, max_measure_len=64):
    is_valid = True
    for measure in segment['note']:
        if len(measure) > max_measure_len:
            is_valid = False
            break
        if tokenizer.has_irregular_token(measure):
            is_valid = False
            break
    return is_valid


def get_normalized_melody(event_file):
    # Load full-score for getting section pickups
    score_event, mark = load_event(event_file)
    event, _ = expand_score(score_event, mark, "no_repeat")

    # Melody extract
    melody = skyline_variation(event)

    # Quantize tempo
    for i in melody:
        melody[i]['tempo'] = normalize_tp(melody[i]['tempo'])

    # Pitch transpose to C major/minor
    melody = normalize_event(melody, adjust_ts_tp=False)
    return melody


def get_base_vocab(event_dir, melody_dir, freq_thresh=50):
    vocab_cnt = Counter()
    ts_set = set()
    tp_set = set()

    # Get melody vocab
    event_file_list = get_file_list(event_dir)
    print("Build Vocabulary")
    for event_file in tqdm(event_file_list):
        melody_file = event_file.replace(event_dir, melody_dir)
        if os.path.exists(melody_file):
            melody = load_note_event(melody_file)
        else:
            melody = get_normalized_melody(event_file)

            # Save pitch transposed, tempo normalized melody
            with open(melody_file, "w") as f:
                json.dump(melody, f)

        # Update vocabulary count
        for i in melody:
            vocab_cnt.update(melody[i]['event'])
            ts_set.add(f"ts-{melody[i]['time_signature']}")
            tp_set.add(f"tp-{melody[i]['tempo']}")

    # Post process vocabulary
    vocab = clean_vocab(vocab_cnt, freq_thresh=freq_thresh)
    base_vocab = sorted(vocab + list(tp_set) + list(ts_set) + ["bar"])
    return base_vocab


def get_dataset(melody_file_list, max_measure_len=64, n_measure=8, hop_measure=2):

    paired_segments = []

    for melody_file in tqdm(melody_file_list):
        # Load pitch transposed, tempo normalized melody
        melody = load_note_event(melody_file)

        # Load tempo/time signature change point file
        cpt_file = melody_file.replace(melody_dir_base, "midi_no_repeat")
        with open(cpt_file) as f:
            cpts = json.load(f)['cpt']
        cpts.append({'measure': max(melody) + 1})
        pickups = get_sect_pickups(melody, cpts[:-1])

        segments = []
        idx_pairs = []
        offset = 0
        for _ in range(0, n_measure, hop_measure):

            cpts[0]['measure'] += hop_measure
            segs = segment(melody, cpts, pickups)
            last_valid = False

            for i, seg in enumerate(segs):
                is_valid = verify_segment(seg, max_measure_len)
                if is_valid:
                    seg_tokens = flatten_measures(seg, max_measure_len)
                    segments.append(tokenizer.convert_tokens_to_ids(seg_tokens))
                else:
                    segments.append(None)
                if last_valid and is_valid:
                    idx_pairs.append([i - 1 + offset, i + offset])

                last_valid = is_valid

            offset += len(segs)

        type_id = tokenizer.token_to_id['next']
        for idx_0, idx_1 in idx_pairs:
            paired_segments.append((segments[idx_0], [type_id] + segments[idx_1]))

    return paired_segments


if __name__ == "__main__":

    data_dir = "../../sonata-dataset-phrase"

    event_dir = os.path.join(data_dir, "event")
    melody_dir_base = "normalized_skyline_no_repeat"
    melody_dir = os.path.join(data_dir, melody_dir_base)
    for composer in COMPOSERS:
        os.makedirs(os.path.join(melody_dir, composer), exist_ok=True)

    # Get Cleaned Vocabulary
    base_vocab_file = os.path.join(data_dir, "vocab", "base_vocab_melody.txt")
    if not os.path.exists(base_vocab_file):
        base_vocab = get_base_vocab(data_dir)
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
    hop_measure = 2
    n_measure = max_seq_len // max_measure_len

    # Get overlapped segments
    out_dir = os.path.join(data_dir, "next_phrase_melody")
    dataset_dir = os.path.join(out_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    print("Get Overlapped Segments")
    print(f"# Measure per segments: {n_measure}")
    print(f"Hop size: {hop_measure} (measures)")

    # Make dataset
    df = pd.read_csv(os.path.join(data_dir, "dataset_split.csv"))
    for split in ['train', 'val']:
        print(f"Process {split} set")
        melody_file_list = [os.path.join(melody_dir, fname)
                            for fname in df[df['split'] == split]['fname']]

        segment_pairs = get_dataset(melody_file_list,
                                    max_measure_len, n_measure, hop_measure)

        with open(os.path.join(dataset_dir, f"{split}_512_pad.json"), "w") as f:
            json.dump(segment_pairs, f)
