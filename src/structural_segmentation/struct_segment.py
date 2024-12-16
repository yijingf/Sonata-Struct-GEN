"""Structural Segmentation

Returns:
    section-phrase structrure
"""

# Todo: 1. Use OrderedDict instead of list!
# Todo: 2. Use KMP to determine score section pattern

import os
import json
import pickle
import numpy as np
from copy import deepcopy
from fractions import Fraction
from collections import OrderedDict

from longest_repeating_pattern import find_lrp

import sys
sys.path.append("..")

from utils.common import load_event, trim_event, token2v
from utils.event import expand_score, no_repeat_pattern
from utils.align import get_t_bar, t_to_bar_beat, bar_beat_to_t


def pattern_type(s):
    if s[0] == 'I':
        s = s[1:]

    if s == 'A':
        return "A", []
    if s == 'AB':
        return "AB", []
    lrp = find_lrp(s)
    if not lrp:
        return "ABC", []
    return 'ABA', [0, len(lrp)]


def get_entropy(A):
    """Shannon entropy

    Args:
        A (np.array): 

    Returns:
        Shannon entropy of np.array A
    """
    pA = A / A.sum()
    Shannon2 = -np.sum(pA * np.log2(pA))
    return Shannon2


def round_to_sect_bound(t_pred, sig_shift, t_thresh=5):
    """Round predicted boundary time to its closest tempo, key/time signature shift time.
    Return [rounded t_pred, True] if a boundary time is found in t_pred +/- t_thresh
    else return [t_pred, False].

    Args:
        t_pred (float): predicted boundary time
        sig_shift (array, optional): _description_. Defaults to None.
        t_thresh (int, optional): _description_. Defaults to 5.
    """
    shift_t = np.array([i['t'] for i in sig_shift])

    t_diff = np.abs(t_pred - shift_t)
    idx = np.argmin(t_diff)
    if t_diff[idx] <= t_thresh:
        return sig_shift[idx]
    return None


def get_signature_shift(unrolled_event, ts_shift):
    """
    Tempo, time signature, key signature changes
    """
    key_signature_shift = []
    prev_key = unrolled_event[0]['key']
    for i in range(max(unrolled_event) + 1):
        if unrolled_event[i]['key'] != prev_key:
            prev_key = unrolled_event[i]['key']
            key_signature_shift.append({"measure": i})

    sig_shift = []
    last_measure = ts_shift[0]['measure'] - 1
    for e in sorted(ts_shift + key_signature_shift, key=lambda x: x['measure']):
        if e['measure'] == last_measure:
            continue
        last_measure = e['measure']

        if 't' in e:
            sig_shift.append(e)
            continue

        t_bar = get_t_bar(**sig_shift[-1])
        t = (e['measure'] - sig_shift[-1]['measure']) * t_bar
        e['t'] = float(sig_shift[-1]['t']) + float(t)
        for key in ['tempo', 'time_signature']:
            e[key] = deepcopy(sig_shift[-1][key])

        sig_shift.append(e)

    return sig_shift


def locate_phrase(phrase_bound, sect_bound, ts_shift, min_phrase=1):
    """Locate the phrases in sections given phrase start/ending time. Modify the boundary in place.

    Args:
        phrase_bound (np.array):
        sect_bound (list): a list of section attributes, i.e.
            [{"name": `Section Name`,
              "interval": [`Section Start Time`, `Section End Time`]}]
        ts_shift (list): _description_
        min_phrase (int): Minimum number of bars per phrase

    Returns:
        _type_: _description_
    """

    sect_phrase = {}
    i = 0
    for name, sect in sect_bound.items():
        phrases = []
        for t_st, t_ed in sect:
            i_st = np.where(phrase_bound[:, 0] == t_st)[0][0]
            i_ed = np.where(phrase_bound[:, 1] == t_ed)[0][0] + 1

            while i < len(ts_shift) - 1:
                if t_st < ts_shift[i + 1]['t']:
                    break
                else:
                    i += 1
            t_thresh = get_t_bar(**ts_shift[i]) * min_phrase

            phrases += merge_intervals(phrase_bound[i_st: i_ed], t_thresh)
        sect_phrase[name] = phrases

    return sect_phrase


def update_phrase_bound(phrase_bound, sect_bound):
    """_summary_

    Args:
        phrase_bound (_type_): _description_
        sect_bound (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_sect_bound = []
    for sect in sect_bound.values():
        t_sect_bound += [e for e in sect]

    t_sect_onset = [i[0] for i in t_sect_bound] + [t_sect_bound[-1][1]]

    t_phrase_onset = np.append(phrase_bound[:, 0], phrase_bound[-1, 1])
    t_phrase_onset = np.sort(np.append(t_phrase_onset, t_sect_onset))
    new_phrase_bound = np.vstack([t_phrase_onset[:-1], t_phrase_onset[1:]]).T
    return new_phrase_bound


def merge_intervals(intervals, t_thresh=1):
    """Merge intervals shorter than `t_thresh` into longer phrases.

    Args:
        intervals (list): A list of phrase start, ending time, i.e. [[t_start, t_end]]
        t_thresh (int, optional): Defaults to 1.

    Returns:
        _type_: _description_
    """
    merged_intervals = []
    last_t_st = intervals[0][0]
    for _, (t_st, t_ed) in enumerate(intervals):
        if round(t_ed - last_t_st) < t_thresh:
            continue
        else:
            merged_intervals.append([last_t_st, t_ed])
            last_t_st = t_ed

    if not len(merged_intervals):
        merged_intervals = [intervals[0]]

    if last_t_st != t_ed:
        merged_intervals[-1][-1] = t_ed

    return merged_intervals


def get_t_primary_theme(sub_sect_bound):
    """_summary_

    Args:
        sub_sect_bound (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Assume primary theme always exists in the beginning of section A on score
    onset_A = sub_sect_bound['A'][0]['st']
    t, measure, beat = onset_A['t'], onset_A['idx'], onset_A['onset']
    return t, measure, beat


def get_delta_measure(beat, std_beat):
    if std_beat < beat:
        return 1
    return 0
    # delta = np.array([-1, 0, 1])
    # delta_idx = np.argmin(delta + std_beat - beat)
    # return delta[delta_idx]


def get_struct_A(sub_sect_bound, boundaries, sig_shift):
    """Map sub-sections on score into music structure. Sections are marked by the onsets of primary themes or tempo, time/key signature shifts.

    Args:
        sub_sect_bound (_type_): _description_
        boundaries (_type_): _description_
        sig_shift (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_boundaries, labels = boundaries
    labels = np.array(labels)

    # Primary theme onset time, measure, beat
    t_A, _, beat_pt = get_t_primary_theme(sub_sect_bound)
    sects = [t_A]

    # Find other occurences of primary theme
    i_pt = np.argmin(np.abs(t_boundaries[:, 0] - t_A))
    t_pts = t_boundaries[labels == labels[i_pt]][:, 0]
    t_pts = t_pts[t_pts > t_boundaries[:, 0][i_pt]]

    # Refine onsets
    for t_pt in t_pts:
        bound = round_to_sect_bound(t_pt, sig_shift)
        if bound:
            sects.append(bound['t'])
        else:
            # Make sure recap starts at the same beat position as expo
            measure, beat = t_to_bar_beat(t_pt, sig_shift)
            measure += get_delta_measure(beat, beat_pt)
            t = bar_beat_to_t((measure, deepcopy(beat_pt)), sig_shift)
            sects.append(float(t))

    # Add signature shift onsets
    sects += [e['t'] for e in sig_shift[1:]]
    # Remove duplicate and sort
    sects = sorted(set(sects))

    # Map to real section
    sect_bound = {"expose": []}
    for i, e in enumerate(sects[:-1]):
        sect_bound['expose'] += [[e, sects[i + 1]]]
    return sect_bound


def get_struct_A_B(sub_sect_bound, boundaries, sig_shift):
    """Map sub-sections on score into sonata structure.
    Exposition is the whole A section on score;
    Development starts with B section;
    Recapitulation starts in the middle of B section if t_recap is not None.

    Args:
        sub_sect_bound (_type_): _description_
        boundaries (_type_): _description_
        sig_shift (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_boundaries, labels = boundaries
    labels = np.array(labels)

    # Exposition
    # Identify primary theme in A
    sect_A = sub_sect_bound['A']
    t_A, _, beat_pt = get_t_primary_theme(sub_sect_bound)

    # Development
    sect_B = sub_sect_bound['B']
    t_dev = sect_B[0]['st']['t']

    # Recapitulation
    # Find second occurrence of primary theme
    i_pt = np.argmin(np.abs(t_boundaries[:, 0] - t_A))
    t_pt = t_boundaries[labels == labels[i_pt]][:, 0]

    if any(t_pt > t_dev):

        i_candidate = 0
        while i_candidate < len(t_pt[t_pt > t_dev]):

            # Try round to closest section boundary
            t_recap = t_pt[t_pt > t_dev][i_candidate]
            bound = round_to_sect_bound(t_recap, sig_shift)
            if bound:
                t_recap = bound['t']
            else:
                # Make sure recap starts at the same beat position as expo
                measure, beat = t_to_bar_beat(t_recap, sig_shift)
                measure += get_delta_measure(beat, beat_pt)
                t_recap = bar_beat_to_t((measure, beat_pt), sig_shift)

            if t_recap > t_dev:
                break

            t_recap = sig_shift[-1]['t']
            i_candidate += 1
    else:
        # Not ABA'
        t_recap = sig_shift[-1]['t']

    sect_bound = {"expose": [], "dev": []}

    # Map sub-sections to real section
    # Exposition
    for e in sect_A:
        sect_bound["expose"] += [[e['st']['t'], e['ed']['t']]]

    # Development
    for i, e in enumerate(sect_B):
        if e['ed']['t'] >= t_recap:
            if e['st']['t'] == t_recap:
                break
            sect_bound['dev'] += [[e['st']['t'], t_recap]]
            break
        else:
            sect_bound["dev"] += [[e['st']['t'], e['ed']['t']]]

    if t_recap == sig_shift[-1]['t']:
        return sect_bound

    # Recapitulation
    sect_bound['recap'] = []
    if e['ed']['t'] > t_recap:
        sect_bound['recap'] += [[t_recap, e['ed']['t']]]

    for e in sect_B[i + 1:]:
        sect_bound['recap'] += [[e['st']['t'], e['ed']['t']]]

    return sect_bound


def get_struct_A_B_A(sub_sect_bound):
    """Find exposition, development and recapitulation in AB-x-AB-(y) form on score.
    Exposition is first A-B section on score;
    Development starts with x section, i.e. C in this corpus;
    Recapitulation starts with A-repeat.

    Args:
        sub_sect_bound (_type_): _description_

    Returns:
        _type_: _description_
    """

    sub_sect_names = list(sub_sect_bound.keys())
    i_dev = sub_sect_names.index('C')
    i_recap = sub_sect_names.index('A-repeat')

    sect_idx = OrderedDict({"expose": [0, i_dev],
                            "dev": [i_dev, i_recap],
                            "recap": [i_recap, len(sub_sect_bound)]})

    # Fill in each section
    sect_bound = {"expose": [], "dev": [], "recap": []}
    for sect_name, idx in sect_idx.items():

        # Iterate through sub-sections
        for name in list(sub_sect_bound)[idx[0]: idx[1]]:

            for e in sub_sect_bound[name]:
                sect_bound[sect_name] += [[e['st']['t'], e['ed']['t']]]

    return sect_bound


def get_struct_A_B_C(sub_sect_bound, boundaries, sig_shift):
    """_summary_

    Args:
        sub_sect_bound (_type_): _description_
        boundaries (_type_): _description_
        ts_shift (_type_): _description_

    Returns:
        _type_: _description_
    """

    t_boundaries, labels = boundaries
    labels = np.array(labels)

    # Exposition
    # Assume exposure always exists in A
    sub_sect_names = list(sub_sect_bound.keys())
    i_expo = sub_sect_names.index('A')
    t_A, _, beat_pt = get_t_primary_theme(sub_sect_bound)

    # Find other occurences of primary theme
    i_pt = np.argmin(np.abs(t_boundaries[:, 0] - t_A))
    t_pts = t_boundaries[labels == labels[i_pt]][:, 0]
    t_pts = t_pts[t_pts > t_boundaries[:, 0][i_pt]]

    # Recapitulation
    i_B = sub_sect_names.index('B')
    t_B = sub_sect_bound['B'][0]['st']['t']
    n_sub_sect = len(sub_sect_bound)

    if any(t_pts > t_B):
        t_recap = t_pts[t_pts > t_B][0]

        # Refine t_recap
        bound = round_to_sect_bound(t_recap, sig_shift)

        if bound:
            t_recap = bound['t']
        else:
            # Make sure recap starts at the same beat position as expo
            measure, beat = t_to_bar_beat(t_recap, sig_shift)
            measure += get_delta_measure(beat, beat_pt)
            t_recap = float(bar_beat_to_t((measure, beat_pt), sig_shift))

        for i in range(i_B, n_sub_sect):
            name = sub_sect_names[i]
            if t_recap < sub_sect_bound[name][-1]['ed']['t']:
                i_recap = i
                break

    else:
        # No recapitulation found
        t_recap, i_recap = sig_shift[-1]['t'], n_sub_sect - 1

    # Development
    if t_recap == sig_shift[-1]['t']:
        i_dev = i_B
    else:
        i_dev = get_i_dev_A_B_C(sub_sect_bound, t_recap, i_expo, i_recap)

    # Fill score subsection boundary into section
    sect_bound = {"expose": [], "dev": []}

    # Exposition
    for name in list(sub_sect_bound)[i_expo: i_dev]:
        for e in sub_sect_bound[name]:
            sect_bound["expose"] += [[e['st']['t'], e['ed']['t']]]

    # Development
    for name in list(sub_sect_bound)[i_dev: i_recap + 1]:
        for e in sub_sect_bound[name]:
            if e['ed']['t'] > t_recap:
                if e['st']['t'] == t_recap:
                    break
                sect_bound["dev"] += [[e['st']['t'], t_recap]]
                break
            else:
                sect_bound["dev"] += [[e['st']['t'], e['ed']['t']]]
                if e['ed']['t'] == t_recap:
                    break

    if t_recap == sig_shift[-1]['t']:
        # recapitulation not found
        return sect_bound

    # Recapitulation
    if e['ed']['t'] > t_recap:
        sect_bound["recap"] = [[t_recap, e['ed']['t']]]

    for e in sub_sect_bound[sub_sect_names[i_recap]][i + 1:]:
        sect_bound["recap"] += [[e['st']['t'], e['ed']['t']]]

    for name in list(sub_sect_bound)[i_recap + 1:]:
        for e in sub_sect_bound[name]:
            sect_bound["recap"] += [[e['st']['t'], e['ed']['t']]]

    return sect_bound


def get_i_dev_A_B_C(sub_sect_bound, t_recap, i_expo, i_recap):
    """
    """

    t_sect_bound = [{"st": float(e[0]['st']['t']),
                     "ed": float(e[-1]['ed']['t'])}
                    for e in sub_sect_bound.values()]

    t_dur = [e['ed'] - e['st'] for e in t_sect_bound][i_expo: i_recap]
    t_dur_recap = sum(t_dur[i_recap - i_expo + 1:])
    t_dur_recap += t_sect_bound[i_recap]['ed'] - t_recap

    if t_recap > t_sect_bound[i_recap]['st']:
        t_dur += [t_recap - t_sect_bound[i_recap]['st'], t_dur_recap]
    else:
        t_dur += [t_dur_recap]

    best_entropy = 0
    i_split = -1

    for i in range(1, len(t_dur) - 1):
        t_split = np.array([sum(t_dur[:i]), sum(t_dur[i:-1]), t_dur[-1]])
        curr_entropy = get_entropy(t_split)
        if curr_entropy > best_entropy:
            i_split = i
            best_entropy = curr_entropy

    i_dev = i_split + i_expo
    return i_dev


def get_t_fin(n_bar, onset, **kwargs):
    """Get time (in seconds) of last bar line.

    Args:
        idx_mapping (dict): _description_
        onset (list): _description_
    """
    n_bar_last_seg = n_bar + 1 - onset[-1]['measure']

    t_last_seg = n_bar_last_seg * get_t_bar(**onset[-1])

    t_fin = onset[-1]['t'] + t_last_seg
    return t_fin


def merge_sub_sect_name(sub_sects):
    """Merge sub-section with same prefix, such as A, A1, A, A2 into one section [A, A1], [A, A2]

    Args:
        sub_sects (list): a list of subsection

    Returns:
        list: list of sub section lists, e.g. [[A, A2], [B, A2]]
    """
    sect, sects = [], []
    last_sub_sect = sub_sects[0]

    for sub_sect in sub_sects:

        if sub_sect[0] != last_sub_sect[0] or len(sub_sect) < len(last_sub_sect):
            sects.append(sect)
            sect = [sub_sect]
        else:
            sect.append(sub_sect)
        last_sub_sect = sub_sect

    if sect:
        sects.append(sect)

    return sects


def get_score_section(struct):
    """Merge score sub-section into sections

    Args:
        struct (_type_): _description_

    Returns:
        _type_: _description_
    """
    sub_sects = no_repeat_pattern(struct['pattern'])
    # Merge volta into section, e.g. A1/A2 -> A
    sects = merge_sub_sect_name(sub_sects)

    # Merge short intro (<4 measure) into adjacent sub-section
    if struct['pattern'][0] == 'I':
        len_I = struct['attr']['A']['idx'] - struct['attr']['I']['idx']
        if len_I < 4:
            sects = sects[1:]
            sects[0] = ['I'] + sects[0]

    if "II" in struct['pattern']:
        II_idx = sects.index(['II'])
        next_sect = sects[II_idx + 1][0]
        len_II = struct['attr'][next_sect]['idx'] - struct['attr']['II']['idx']
        if len_II < 4:
            sects = sects[:II_idx] + sects[II_idx + 1:]
            sects[II_idx] = ['II'] + sects[II_idx]

    if len(sects) > 1 and sects[0] == ['A']:
        next_sect = sects[1][0]
        if struct['attr'][next_sect]['idx'] - struct['attr']['A']['idx'] < 4:
            sects = sects[1:]
            sects[0] = ['A'] + sects[0]
    return sects


def get_score_sub_sect_boundary(struct, n_measure):
    sub_sect_onset = sorted([(i, v) for i, v in struct['attr'].items()],
                            key=lambda x: (x[1]['idx'], x[1]['onset']))
    sub_sect_onset.append(('Fin', {"idx": n_measure + 1, "onset": "o-0"}))

    sub_sect_bound = {}
    for i, (sub_sect_name, onset) in enumerate(sub_sect_onset[:-1]):
        sub_sect_bound[sub_sect_name] = {"st": onset,
                                         "ed": sub_sect_onset[i + 1][-1]}
    return sub_sect_bound


def get_midi_sect_boundary(pattern, sects, struct, score_sub_sect_bound, n_measure):
    """Converting score sections to midi sections

    Args:
        pattern (_type_): _description_
        sects (_type_): _description_
        struct (_type_): _description_
        score_sub_sect_bound (_type_): _description_
        n_measure (int): number of measures in midi

    Returns:
        _type_: _description_
    """
    i_measure = 0
    midi_sect_onset = {}
    is_first_repeat = False

    # Unroll pattern
    for i_sect, sect_name in enumerate(pattern):

        sub_sect_name = sects[i_sect][0]

        if sect_name in midi_sect_onset:
            sect_name = f'{sect_name}-repeat'
            is_first_repeat = not (is_first_repeat)

        pos = struct['attr'][sub_sect_name]['onset']
        if pos != 'o-0' and is_first_repeat:
            i_measure -= 1
        midi_sect_onset[sect_name] = {"idx": i_measure, "onset": token2v(pos)}

        for sub_sect_name in sects[i_sect]:
            bound = score_sub_sect_bound[sub_sect_name]
            i_measure += bound['ed']['idx'] - bound['st']['idx']

    midi_sect_onset = sorted([(i, v) for i, v in midi_sect_onset.items()],
                             key=lambda x: (x[1]['idx'], x[1]['onset']))
    midi_sect_onset.append(('Fin', {"idx": n_measure + 1, "onset": 0}))

    midi_sect_bound = OrderedDict()
    for i, (sect_name, onset) in enumerate(midi_sect_onset[:-1]):
        midi_sect_bound[sect_name] = {"st": onset,
                                      "ed": midi_sect_onset[i + 1][-1]}
    return midi_sect_bound


def get_sub_sect_boundary(sig_shift, midi_sect_bound):

    # Revise the last score section boundary
    beat = Fraction(sig_shift[-1]['time_signature']) * 4
    sect_name = list(midi_sect_bound.keys())[-1]
    n_measure = midi_sect_bound[sect_name]['ed']['idx'] - 1
    midi_sect_bound[sect_name]['ed'] = {"idx": n_measure, "onset": beat}

    # Combine repeat signs with time/key signature and tempo shifts
    bounds = [(i['st']['idx'], i['st']['onset'])
              for i in midi_sect_bound.values()]
    bounds += [(i['measure'], 0) for i in sig_shift[:-1]]
    bounds = sorted(set(bounds))

    sub_sect_bound = OrderedDict()
    for name, e in midi_sect_bound.items():
        st = (e['st']['idx'], e['st']['onset'])
        ed = (e['ed']['idx'], e['ed']['onset'])
        seg = [st] + [i for i in bounds if i < ed and i > st] + [ed]
        seg = [(i[0], i[1], float(bar_beat_to_t(i, sig_shift))) for i in seg]
        sub_sect_bound[name] = []
        for i, pos in enumerate(seg[:-1]):
            sub_sect_bound[name] += [{"st": {"idx": pos[0], "onset": pos[1], "t": pos[2]},
                                      "ed": {"idx": seg[i + 1][0],
                                             "onset": seg[i + 1][1],
                                             "t": seg[i + 1][2]}}]
    return sub_sect_bound


def get_pattern(sects):
    pattern = []
    for i, sect in enumerate(sects):
        if sect[0] == "I" and i == 0 and len(sect) > 1:
            pattern += [sect[1]]
        elif sect[0] == "II" and len(sect) > 1:
            pattern += [sect[1]]
        else:
            pattern += [sect[0]]

    # Rename
    exist_name = set()

    renamed = [pattern[0]]
    last_name = pattern[0]
    for name in pattern[1:]:
        if last_name[0] == 'I' or ord(name) < ord(last_name):
            pass
        elif ord(name) - ord(last_name) > 1:
            new_name = chr(ord(last_name) + 1)
            if new_name in exist_name:
                new_name = chr(ord(max(exist_name)) + 1)
            name = new_name
        renamed.append(name)

        if name != 'I':
            exist_name.add(name)

        last_name = name
    return renamed


def main(composer, file_base, config, n_cluster=-1, min_phrase=4):

    # Load event file
    event_file = os.path.join(config.event_dir, composer, f"{file_base}.json")
    score_event, struct = load_event(event_file)
    event, idx_mapping = expand_score(score_event, struct, "no_repeat")

    sects = get_score_section(struct)
    pattern = get_pattern(sects)
    pattern_str = "-".join(pattern)

    # Sub-section boundary on score
    score_sect_bound = get_score_sub_sect_boundary(struct, max(score_event))

    # Section boundary on midi
    midi_sect_bound = get_midi_sect_boundary(pattern, sects, struct,
                                             score_sect_bound, max(idx_mapping))

    # Load score measure to midi measure mapping
    map_file = os.path.join(config.midi_dir, composer, f"{file_base}.json")
    with open(map_file) as f:
        ts_shift = json.load(f)['onset']

    # Renew end time
    t_fin = get_t_fin(max(idx_mapping), ts_shift)
    ts_shift.append({"measure": max(idx_mapping) + 1, "t": float(t_fin),
                     "tempo": ts_shift[-1]['tempo'],
                     "time_signature": ts_shift[-1]['time_signature']})

    # Split sections by tempo, time/key signature shifts, for refining phrase boundaries
    sig_shift = get_signature_shift(event, ts_shift)
    sub_sect_bound = get_sub_sect_boundary(sig_shift, midi_sect_bound)

    # Load predicted boundaries
    boundary_file = os.path.join(config.boundary_dir,
                                 f"{composer}-{file_base}.pkl")
    with open(boundary_file, "rb") as f:
        pred_bound = pickle.load(f)

    # Find section boundary based on score section and structural boundary prediction
    intro = None
    if pattern_str[0] == 'I':
        intro_sect = deepcopy(sub_sect_bound['I'])
        intro = [[e['st']['t'], e['ed']['t']] for e in intro_sect]
        pattern_str = pattern_str[2:]

    # Longest Common Prefix (Kasai)
    seg_bound = pred_bound[n_cluster]
    if pattern_str == 'A':
        sect_bound = get_struct_A(sub_sect_bound, seg_bound, sig_shift)

    elif pattern_str == 'A-B':
        sect_bound = get_struct_A_B(sub_sect_bound, seg_bound, sig_shift)

    elif len(pattern_str.split('A-B')) > 2:
        # AB-x-AB or AB-x-AB-y
        sect_bound = get_struct_A_B_A(sub_sect_bound)
    else:
        # A-B-A' or A-B-C type
        sect_bound = get_struct_A_B_C(sub_sect_bound, seg_bound, sig_shift)

    if intro is not None:
        # Todo: consider using ordereddict
        sect_bound = {"intro": intro, **sect_bound}

    # Locate phrases in section
    phrase_bound = update_phrase_bound(pred_bound[n_cluster][0], sect_bound)
    sect_phrase = locate_phrase(phrase_bound, sect_bound, ts_shift, min_phrase)
    sect_phrase_pos = {}

    sect_event = {}
    res_key = ['tempo', 'time_signature', 'key']

    for name, t_phrases in sect_phrase.items():
        sect_phrase_pos[name] = []
        phrases = []
        for t_phrase in t_phrases:
            st_pos = t_to_bar_beat(t_phrase[0], sig_shift)
            ed_pos = t_to_bar_beat(t_phrase[1], sig_shift)

            # Todo: Round position to beat
            phrase = trim_event(event, st_pos, ed_pos)

            if not len(phrase):
                print(composer, file_base)
                continue

            sect_phrase_pos[name].append([(st_pos[0], str(st_pos[1])),
                                          (ed_pos[0], str(ed_pos[1]))])

            i_st, i_ed = min(phrase), max(phrase)
            item = {k: phrase[i_st][k] for k in res_key}
            item['event'] = [phrase[i]['event'] for i in range(i_st, i_ed + 1)]

            phrases += [item]

        sect_event[name] = phrases

    return sect_event, sect_phrase_pos


if __name__ == "__main__":
    import argparse
    from glob import glob

    class config:
        data_dir = "../../sonata-dataset-phrase"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default=config.data_dir, help="Path to dataset")

    args = parser.parse_args()

    config.data_dir = args.data_dir
    config.event_dir = os.path.join(args.data_dir, "event")
    config.midi_dir = os.path.join(args.data_dir, "rendered_midi_no_repeat")
    config.boundary_dir = os.path.join(args.data_dir, "boundary_predictions")

    for composer in ['mozart', 'beethoven', 'scarlatti', 'haydn']:
        file_list = glob(os.path.join(config.event_dir, f"{composer}/*.json"))

        for event_file in sorted(file_list):
            file_base = os.path.basename(event_file).split(".")[0]

            try:
                sect_phrase, sect_phrase_pos = main(
                    composer, file_base, config)
                with open(os.path.join(config.data_dir, "struct", f"{composer}-{file_base}.json"),
                          "w") as f:
                    json.dump(sect_phrase, f)

                with open(os.path.join(config.data_dir, "struct_mark", f"{composer}-{file_base}.json"), "w") as f:
                    json.dump(sect_phrase_pos, f)
            except:
                print(f"Error: {composer}-{file_base}")
