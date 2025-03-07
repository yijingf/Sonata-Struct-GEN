# See norm_event_for_transformer_input.ipynb for manual processing

import os
import json
import numpy as np
from fractions import Fraction

import sys
sys.path.append("..")

from utils.event import Event
from utils.decode import decode_token_to_event
from utils.common import get_file_list, token2v, load_note_event


def is_irregular(div):
    if not div % 5 or not div % 11 or not div % 7:
        return True


def has_irregular_div(tokens):
    for token in tokens:
        if token[0] in ['o', 'd']:
            div = token2v(token).denominator
            if is_irregular(div):
                return True
    return False


def closest_multiple(n):
    log_n = int(np.log2(n))
    f_3 = 2 ** (log_n - 1) * 3
    f_2 = 2 ** log_n
    if f_3 < n:
        return f_3
    else:
        return f_2


def norm_note_div(seq):
    onset = seq[0][0]
    div = max([i[-1].denominator for i in seq])
    norm_div = closest_multiple(div)
    bins = np.array([Fraction(i, norm_div) for i in range(norm_div)])

    if is_irregular(onset.denominator):
        idx = np.argmin(np.abs(onset - bins))
        onset = Fraction(bins[idx])

    offset = seq[-1][0] + seq[-1][-1]

    norm_seq = []
    for on, pitch, dur in seq:
        new_on = bins[np.argmin(np.abs(on - onset - bins))] + onset
        new_dur = Fraction(dur * div, norm_div)
        if new_on + new_dur > offset:
            break
        if not len(norm_seq) or new_on > norm_seq[-1][0]:
            norm_seq.append([new_on, pitch, new_dur])
    return norm_seq


def norm_div_tokens(tokens, dummy_ts_tp=["ts-4/4", "tp-120"]):
    flag = False
    notes = decode_token_to_event(dummy_ts_tp + tokens)[-1][0]
    normed_notes = []
    irr_seq = []
    for on, pitch, dur in notes:
        if is_irregular(on.denominator) or is_irregular(dur.denominator):
            flag = True
            if not len(irr_seq) or irr_seq[-1][-1] + irr_seq[-1][0] == on:
                irr_seq.append([on, pitch, dur])
            else:
                normed_notes = norm_note_div(irr_seq)
                irr_seq = [[on, pitch, dur]]
        else:
            normed_notes.append([on, pitch, dur])

    if len(irr_seq):
        normed_notes += norm_note_div(irr_seq)
        irr_seq = []

    if not flag:
        return tokens

    normed_tokens = []
    for on, pitch, dur in normed_notes:
        normed_tokens += [f"o-{on}", pitch, f"d-{dur}"]

    return normed_tokens


def merge_div(on, dur, div, div_dict={24: 12, 32: 8, 16: 8}):
    normed_div = div_dict[div]
    bins = np.arange(0, normed_div) * Fraction(1, normed_div)
    on_idx = np.argmin(np.abs(on - int(on) - bins))
    dur_idx = np.argmin(np.abs(dur - bins))
    new_on = int(on) + bins[on_idx]
    new_dur = max(bins[dur_idx], Fraction(1, normed_div))
    return new_on, new_dur


DATA_DIR = "../../sonata-dataset-phrase"
melody_dir = os.path.join(DATA_DIR, "skyline_no_repeat")
normed_melody_dir = os.path.join(DATA_DIR, "norm_skyline_no_repeat")

acc_dir = os.path.join(DATA_DIR, "acc_no_repeat")
normed_acc_dir = os.path.join(DATA_DIR, "norm_acc_no_repeat")


# 1. Quantize irregular rhythm 1/5, 1/7, 1/11
orig_melody_event_files = sorted(get_file_list(melody_dir))
orig_acc_event_files = sorted(get_file_list(acc_dir))
for event_file in orig_melody_event_files + orig_acc_event_files:

    event = load_note_event(event_file)
    for i in event:
        tokens = event[i]['event']
        if has_irregular_div(tokens):
            event[i]['event'] = norm_div_tokens(tokens)

    composer, basename = event_file.split(".json")[0].split("/")[-2:]
    if "acc_no_repeat" in event_file:
        output_file = os.path.join(normed_acc_dir, composer, f"{basename}.json")
    else:
        output_file = os.path.join(normed_melody_dir, composer, f"{basename}.json")
    with open(output_file, "w") as f:
        json.dump(event, f)


# 2. Quantize small division: 1/24 -> 1/12; 1/16, 1/32 -> 1/8
dummy_ts_tp = ["ts-4/4", "tp-120"]
melody_event_files = get_file_list(normed_melody_dir)

for event_file in melody_event_files:
    flag = False
    event = load_note_event(event_file)

    for i in event:
        measure_flag = False
        notes = decode_token_to_event(dummy_ts_tp + event[i]['event'])[-1][0]
        normed_notes = []

        for on, pitch, dur in notes:
            div = max(on.denominator, dur.denominator)
            if div >= 16:
                flag, measure_flag = True, True
                on, dur = merge_div(on, dur, div)
                if not len(normed_notes) or on > normed_notes[-1][0]:
                    normed_notes.append([on, pitch, dur])
            else:
                normed_notes.append([on, pitch, dur])

        if measure_flag:
            normed_tokens = []
            for on, pitch, dur in normed_notes:
                normed_tokens += [f"o-{on}", pitch, f"d-{dur}"]
            event[i]['event'] = normed_tokens

    if flag:
        with open(event_file, "w") as f:
            json.dump(event, f)

acc_event_files = get_file_list(normed_acc_dir)
for event_file in acc_event_files:
    flag = False
    event = load_note_event(event_file)

    for i in event:
        has_chord, measure_flag = False, False
        notes = decode_token_to_event(dummy_ts_tp + event[i]['event'])[-1][0]
        normed_notes = []
        irr_on = []
        for on, pitch, dur in notes:
            div = max(on.denominator, dur.denominator)
            if div >= 16:
                if not len(irr_on):
                    irr_on += [on]
                elif on == irr_on[-1]:
                    has_chord = True

                flag, measure_flag = True, True
                on, dur = merge_div(on, dur, div)
                if not len(normed_notes) or on > normed_notes[-1][0]:
                    normed_notes.append([on, pitch, dur])
            else:
                normed_notes.append([on, pitch, dur])

        if not has_chord:
            if measure_flag:
                normed_tokens = []
                for on, pitch, dur in normed_notes:
                    normed_tokens += [f"o-{on}", pitch, f"d-{dur}"]
                event[i]['event'] = normed_tokens
        else:
            print(f"{event_file} measure {i} has chords, manually process this file.")
            continue

    if flag:
        with open(event_file, "w") as f:
            json.dump(event, f)

# 3. Trim extra long duration (> 8 beat) to 8 beat
for event_file in melody_event_files + acc_event_files:
    flag = False
    event = load_note_event(event_file)

    for i in event:
        measure_flag = False
        notes = decode_token_to_event(dummy_ts_tp + event[i]['event'])[-1][0]

        for j, (on, pitch, dur) in enumerate(notes):
            if dur > 8:
                measure_flag, flag = True, True
                notes[j][-1] = Fraction(8, 1)

        if measure_flag:
            normed_tokens = []
            for on, pitch, dur in notes:
                normed_tokens += [f"o-{on}", pitch, f"d-{dur}"]
            event[i]['event'] = normed_tokens

    if flag:
        with open(event_file, "w") as f:
            json.dump(event, f)

# 4. Normalize low frequency duration

from collections import Counter
vocab_cnt = Counter()

for event_file in melody_event_files + acc_event_files:
    event = load_note_event(event_file)
    for i in event:
        vocab_cnt.update(event[i]['event'])

tokens = list(vocab_cnt.keys())
for token in tokens:
    if token[0] != 'd':
        vocab_cnt.pop(token)

low_freq_d = [i for i, v in vocab_cnt.items() if v < 10]

low_freq_d_dict = {}
for token in low_freq_d:
    v = Event(token).val
    norm_token = f"d-{round(v)}"
    if norm_token not in vocab_cnt:
        print(token)
    else:
        low_freq_d_dict[token] = norm_token

low_freq_d_dict['d-7/12'] = 'd-1/2'

for event_file in melody_event_files + acc_event_files:
    event = load_note_event(event_file)
    flag = False
    for i in event:
        for j, token in enumerate(event[i]['event']):
            if token in low_freq_d_dict:
                flag = True
                event[i]['event'][j] = low_freq_d_dict[token]
    if flag:
        with open(event_file, "w") as f:
            json.dump(event, f)

# 5. Normalize tempo and Transpose pitch to C major/minor
from utils.common import normalize_event, normalize_tp

for event_file in acc_event_files + melody_event_files:
    event = load_note_event(event_file)

    # Quantize tempo
    for i in event:
        event[i]['tempo'] = normalize_tp(event[i]['tempo'])

    # Pitch transpose to C major/minor
    event = normalize_event(event, adjust_ts_tp=False)
    with open(event_file, "w") as f:
        json.dump(event, f)
