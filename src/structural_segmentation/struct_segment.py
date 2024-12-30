"""Structural Segmentation

Returns:
    section-phrase structrure
"""

# Todo: 1. Use OrderedDict instead of list!

import os
import json
import pickle
import warnings
import numpy as np
from copy import deepcopy
from fractions import Fraction
from collections import OrderedDict

import sys
sys.path.append("..")
from utils.common import load_event, trim_event
from utils.event import expand_score, no_repeat_pattern
from utils.align import get_t_bar, t_to_bar_pos, bar_pos_to_t
from structural_segmentation.longest_repeating_pattern import find_lrp
from structural_segmentation.struct_label import get_struct_label


def get_pattern_type(s):
    if s[0] == "I":
        s = s[1:]

    if s == 'A':
        return 'A', None
    if s == "AB":
        return "AB", None
    lrp = find_lrp(s)
    if not lrp:
        return "ABC", None
    return "ABA", s[len(lrp)]


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


def round_to_sect_bound(t_pred, sig_cpt, t_thresh=5):
    """Round predicted boundary time to its closest tempo, key/time signature shift time.
    Return [rounded t_pred, True] if a boundary time is found in t_pred +/- t_thresh
    else return [t_pred, False].

    Args:
        t_pred (float): predicted boundary time
        sig_cpt (array, optional): _description_. Defaults to None.
        t_thresh (int, optional): _description_. Defaults to 5.
    """
    shift_t = np.array([i['t'] for i in sig_cpt])

    t_diff = np.abs(t_pred - shift_t)
    idx = np.argmin(t_diff)
    if t_diff[idx] <= t_thresh:
        return sig_cpt[idx]
    return None


def get_reverse_idx_mapping(idx_mapping):
    reversed_idx_mapping = []
    for midi_idx, score_idxs in idx_mapping.items():
        reversed_idx_mapping += [(v, midi_idx) for v in score_idxs]
    return reversed_idx_mapping


def get_expanded_key_cpt(key_cpt, idx_mapping):
    expanded_key_cpt = []

    score_key_cpt_idx = {v['measure']: i for i, v in enumerate(key_cpt)}
    reversed_idx_mapping = get_reverse_idx_mapping(idx_mapping)

    for idx_score, idx_midi in reversed_idx_mapping:
        i = score_key_cpt_idx.get(idx_score, None)
        if i is not None:
            expanded_key_cpt.append({"measure": idx_midi,
                                     "pos": key_cpt[i]['pos'],
                                     "key": key_cpt[i]['key']})
    return expanded_key_cpt


def get_signature_cpt(key_cpt, ts_cpt):
    """
    Tempo, time signature, key signature changes
    """
    sig_cpt = []
    last_measure = ts_cpt[0]['measure'] - 1

    for e in sorted(ts_cpt + key_cpt, key=lambda x: x['measure']):
        if e['measure'] == last_measure:
            continue

        last_measure = e['measure']

        if 't' in e:
            if 'pos' not in e:
                e['pos'] = 0
            sig_cpt.append(e)
            continue

        t_bar = get_t_bar(**sig_cpt[-1])
        t = (e['measure'] - sig_cpt[-1]['measure']) * t_bar
        t -= sig_cpt[-1]['pos'] * 60 / sig_cpt[-1]['tempo']
        e['t'] = float(sig_cpt[-1]['t']) + float(t)
        if 'pos' in e:
            e['pos'] = Fraction(e['pos'])
            e['t'] += float(e['pos'] * 60 / sig_cpt[-1]['tempo'])
        else:
            e['pos'] = 0

        for key in ['tempo', 'time_signature']:
            e[key] = deepcopy(sig_cpt[-1][key])

        sig_cpt.append(e)

    return sig_cpt


def locate_phrase(phrase_bound, sect_bound, ts_cpt, min_phrase=1):
    """Locate the phrases in sections given phrase start/ending time. Modify the boundary in place.

    Args:
        phrase_bound (np.array):
        sect_bound (list): a list of section attributes, i.e.
            [{"name": `Section Name`,
              "interval": [`Section Start Time`, `Section End Time`]}]
        ts_cpt (list): _description_
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

            while i < len(ts_cpt) - 1:
                if t_st < ts_cpt[i + 1]['t']:
                    break
                else:
                    i += 1
            t_thresh = get_t_bar(**ts_cpt[i]) * min_phrase

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
    t, measure, beat = onset_A['t'], onset_A["measure"], onset_A["pos"]
    return t, measure, beat


def get_delta_measure(beat, std_beat):
    if std_beat < beat:
        return 1
    return 0
    # delta = np.array([-1, 0, 1])
    # delta_idx = np.argmin(delta + std_beat - beat)
    # return delta[delta_idx]


def get_struct_A(sub_sect_bound, boundaries, sig_cpt):
    """Map sub-sections on score into music structure. Sections are marked by the onsets of primary themes or tempo, time/key signature shifts.

    Args:
        sub_sect_bound (_type_): _description_
        boundaries (_type_): _description_
        sig_cpt (_type_): _description_

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
        bound = round_to_sect_bound(t_pt, sig_cpt)
        if bound:
            sects.append(bound['t'])
        else:
            # Make sure recap starts at the same beat position as expo
            measure, beat = t_to_bar_pos(t_pt, sig_cpt)
            measure += get_delta_measure(beat, beat_pt)
            t = bar_pos_to_t((measure, deepcopy(beat_pt)), sig_cpt)
            sects.append(float(t))

    # Add signature shift onsets
    sects += [e['t'] for e in sig_cpt[1:]]
    # Remove duplicate and sort
    sects = sorted(set(sects))

    # Map to real section
    sect_bound = {'expose': []}
    for i, e in enumerate(sects[:-1]):
        sect_bound['expose'] += [[e, sects[i + 1]]]
    return sect_bound


def get_struct_A_B(sub_sect_bound, boundaries, sig_cpt):
    """Map sub-sections on score into sonata structure.
    Exposition is the whole A section on score;
    Development starts with B section;
    Recapitulation starts in the middle of B section if t_recap is not None.

    Args:
        sub_sect_bound (_type_): _description_
        boundaries (_type_): _description_
        sig_cpt (_type_): _description_

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
            bound = round_to_sect_bound(t_recap, sig_cpt)
            if bound:
                t_recap = bound['t']
            else:
                # Make sure recap starts at the same beat position as expo
                measure, beat = t_to_bar_pos(t_recap, sig_cpt)
                measure += get_delta_measure(beat, beat_pt)
                t_recap = bar_pos_to_t((measure, beat_pt), sig_cpt)

            if t_recap > t_dev:
                break

            t_recap = sig_cpt[-1]['t']
            i_candidate += 1
    else:
        # Not ABA'
        t_recap = sig_cpt[-1]['t']

    sect_bound = {'expose': [], 'dev': []}

    # Map sub-sections to real section
    # Exposition
    for e in sect_A:
        sect_bound['expose'] += [[e['st']['t'], e['ed']['t']]]

    # Development
    for i, e in enumerate(sect_B):
        if e['ed']['t'] >= t_recap:
            if e['st']['t'] == t_recap:
                break
            sect_bound['dev'] += [[e['st']['t'], t_recap]]
            break
        else:
            sect_bound['dev'] += [[e['st']['t'], e['ed']['t']]]

    if t_recap == sig_cpt[-1]['t']:
        return sect_bound

    # Recapitulation
    sect_bound['recap'] = []
    if e['ed']['t'] > t_recap:
        sect_bound['recap'] += [[t_recap, e['ed']['t']]]

    for e in sect_B[i + 1:]:
        sect_bound['recap'] += [[e['st']['t'], e['ed']['t']]]

    return sect_bound


def get_struct_A_B_A(sub_sect_bound, dev_sect='C'):
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
    i_dev = sub_sect_names.index(dev_sect)
    i_recap = sub_sect_names.index('A-repeat')

    sect_idx = OrderedDict({'expose': [0, i_dev],
                            'dev': [i_dev, i_recap],
                            'recap': [i_recap, len(sub_sect_bound)]})

    # Fill in each section
    sect_bound = {'expose': [], 'dev': [], 'recap': []}
    for sect_name, idx in sect_idx.items():

        # Iterate through sub-sections
        for name in list(sub_sect_bound)[idx[0]: idx[1]]:

            for e in sub_sect_bound[name]:
                sect_bound[sect_name] += [[e['st']['t'], e['ed']['t']]]

    return sect_bound


def get_struct_A_B_C(sub_sect_bound, boundaries, sig_cpt):
    """_summary_

    Args:
        sub_sect_bound (_type_): _description_
        boundaries (_type_): _description_
        ts_cpt (_type_): _description_

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
        bound = round_to_sect_bound(t_recap, sig_cpt)

        if bound:
            t_recap = bound['t']
        else:
            # Make sure recap starts at the same beat position as expo
            measure, beat = t_to_bar_pos(t_recap, sig_cpt)
            measure += get_delta_measure(beat, beat_pt)
            t_recap = float(bar_pos_to_t((measure, beat_pt), sig_cpt))

        for i in range(i_B, n_sub_sect):
            name = sub_sect_names[i]
            if t_recap < sub_sect_bound[name][-1]['ed']['t']:
                i_recap = i
                break

    else:
        # No recapitulation found
        t_recap, i_recap = sig_cpt[-1]['t'], n_sub_sect - 1

    # Development
    if t_recap == sig_cpt[-1]['t']:
        i_dev = i_B
    else:
        i_dev = get_i_dev_A_B_C(sub_sect_bound, t_recap, i_expo, i_recap)

    # Fill score subsection boundary into section
    sect_bound = {'expose': [], 'dev': []}

    # Exposition
    for name in list(sub_sect_bound)[i_expo: i_dev]:
        for e in sub_sect_bound[name]:
            sect_bound['expose'] += [[e['st']['t'], e['ed']['t']]]

    # Development
    for name in list(sub_sect_bound)[i_dev: i_recap + 1]:
        for e in sub_sect_bound[name]:
            if e['ed']['t'] > t_recap:
                if e['st']['t'] == t_recap:
                    break
                sect_bound['dev'] += [[e['st']['t'], t_recap]]
                break
            else:
                sect_bound['dev'] += [[e['st']['t'], e['ed']['t']]]
                # if e['ed']['t'] == t_recap:
                #     break

    if t_recap == sig_cpt[-1]['t']:
        # recapitulation not found
        return sect_bound

    # Recapitulation
    if e['ed']['t'] > t_recap:
        sect_bound['recap'] = [[t_recap, e['ed']['t']]]
    # elif e['ed']['t'] == t_recap:
    #     e = sub_sect_bound[sub_sect_names[i_recap]][i]
    #     sect_bound['recap'] = [[t_recap, e['ed']['t']]]

    for e in sub_sect_bound[sub_sect_names[i_recap]][i + 1:]:
        sect_bound['recap'] += [[e['st']['t'], e['ed']['t']]]

    for name in list(sub_sect_bound)[i_recap + 1:]:
        for e in sub_sect_bound[name]:
            sect_bound['recap'] += [[e['st']['t'], e['ed']['t']]]

    return sect_bound


def get_i_dev_A_B_C(sub_sect_bound, t_recap, i_expo, i_recap):
    """
    """

    t_sect_bound = [{'st': float(e[0]['st']['t']), 'ed': float(e[-1]['ed']['t'])}
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


def get_score_section(mark):
    """Merge score sub-section into sections

    Args:
        mark (_type_): _description_

    Returns:
        _type_: _description_
    """
    sub_sects = no_repeat_pattern(mark["pattern"])
    # Merge volta into section, e.g. A1/A2 -> A
    sects = merge_sub_sect_name(sub_sects)

    # Merge short intro (<4 measure) into adjacent sub-section
    if mark["pattern"][0] == "I":
        len_I = mark["sect"]['A']["measure"] - mark["sect"]["I"]["measure"]
        if len_I < 4:
            sects = sects[1:]
            sects[0] = ["I"] + sects[0]

    if "II" in mark["pattern"]:
        II_idx = sects.index(["II"])
        next_sect = sects[II_idx + 1][0]
        len_II = mark["sect"][next_sect]["measure"] - mark["sect"]["II"]["measure"]
        if len_II < 4:
            sects = sects[:II_idx] + sects[II_idx + 1:]
            sects[II_idx] = ["II"] + sects[II_idx]

    if len(sects) > 1 and sects[0] == ['A']:
        next_sect = sects[1][0]
        if mark["sect"][next_sect]["measure"] - mark["sect"]['A']["measure"] < 4:
            sects = sects[1:]
            sects[0] = ['A'] + sects[0]
    return sects


def get_score_sub_sect_boundary(struct, n_measure):
    sub_sect_onset = sorted([(i, v) for i, v in struct["sect"].items()],
                            key=lambda x: (x[1]["measure"], x[1]["pos"]))
    sub_sect_onset.append(("Fin", {"measure": n_measure + 1, "pos": "0"}))

    sub_sect_bound = {}
    for i, (sub_sect_name, onset) in enumerate(sub_sect_onset[:-1]):
        sub_sect_bound[sub_sect_name] = {'st': onset,
                                         'ed': sub_sect_onset[i + 1][-1]}
    return sub_sect_bound


def get_midi_sect_boundary(pattern, sects, mark, score_sub_sect_bound, n_measure):
    """Converting score sections to midi sections

    Args:
        pattern (_type_): _description_
        sects (_type_): _description_
        mark (_type_): _description_
        score_sub_sect_bound (_type_): _description_
        n_measure (int): number of measures in midi

    Returns:
        _type_: _description_
    """
    i_measure = 0
    midi_sect_onset = {}
    is_first_repeat = False

    # Unfold pattern
    for i_sect, sect_name in enumerate(pattern):

        sub_sect_name = sects[i_sect][0]

        if sect_name in midi_sect_onset:
            sect_name = f"{sect_name}-repeat"
            is_first_repeat = not (is_first_repeat)

        pos = Fraction(mark["sect"][sub_sect_name]["pos"])
        if pos != 0 and is_first_repeat:
            i_measure -= 1
        midi_sect_onset[sect_name] = {"measure": i_measure, "pos": pos}

        for sub_sect_name in sects[i_sect]:
            bound = score_sub_sect_bound[sub_sect_name]
            i_measure += bound['ed']["measure"] - bound['st']["measure"]

    midi_sect_onset = sorted([(i, v) for i, v in midi_sect_onset.items()],
                             key=lambda x: (x[1]["measure"], x[1]["pos"]))
    midi_sect_onset.append(("Fin", {"measure": n_measure, "pos": 0}))

    midi_sect_bound = OrderedDict()
    for i, (sect_name, onset) in enumerate(midi_sect_onset[:-1]):
        midi_sect_bound[sect_name] = {'st': onset,
                                      'ed': midi_sect_onset[i + 1][-1]}
    return midi_sect_bound


def get_sub_sect_boundary(sig_cpt, midi_sect_bound):

    # Revise the last score section boundary
    pos = Fraction(sig_cpt[-1]["time_signature"]) * 4
    sect_name = list(midi_sect_bound.keys())[-1]
    n_measure = midi_sect_bound[sect_name]['ed']["measure"] - 1
    midi_sect_bound[sect_name]['ed'] = {"measure": n_measure, "pos": pos}

    # Combine repeat signs with time/key signature and tempo shifts
    bounds = [(i['st']["measure"], i['st']["pos"]) for i in midi_sect_bound.values()]
    bounds += [(i["measure"], 0) for i in sig_cpt[:-1]]
    bounds = sorted(set(bounds))

    sub_sect_bound = OrderedDict()
    for name, e in midi_sect_bound.items():
        st = (e['st']["measure"], e['st']["pos"])
        ed = (e['ed']["measure"], e['ed']["pos"])
        seg = [st] + [i for i in bounds if i < ed and i > st] + [ed]
        seg = [(i[0], i[1], float(bar_pos_to_t(i, sig_cpt))) for i in seg]
        sub_sect_bound[name] = []
        for i, pos in enumerate(seg[:-1]):
            sub_sect_bound[name] += [{'st': {"measure": pos[0], "pos": pos[1], 't': pos[2]},
                                      'ed': {"measure": seg[i + 1][0], "pos": seg[i + 1][1],
                                             't': seg[i + 1][2]}}]
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
        if last_name[0] == "I" or ord(name) < ord(last_name):
            pass
        elif ord(name) - ord(last_name) > 1:
            new_name = chr(ord(last_name) + 1)
            if new_name in exist_name:
                new_name = chr(ord(max(exist_name)) + 1)
            name = new_name
        renamed.append(name)

        if name != "I":
            exist_name.add(name)

        last_name = name
    return renamed


def main(composer, basename, dirConfig, n_cluster=5, min_phrase=4):

    # Load score measure to midi measure mapping
    map_file = os.path.join(dirConfig.midi_dir, composer, f"{basename}.json")
    with open(map_file) as f:
        ts_cpt = json.load(f)["cpt"]
    if len(ts_cpt) > 1:
        print(f"{composer}-{basename}: multiple tempo/time signature shifts")
        return

    # Load event file
    event_file = os.path.join(dirConfig.event_dir, composer, f"{basename}.json")
    score_event, mark = load_event(event_file)
    event, idx_mapping = expand_score(score_event, mark, "no_repeat")

    sects = get_score_section(mark)
    pattern = get_pattern(sects)

    # Sub-section boundary on score
    score_sect_bound = get_score_sub_sect_boundary(mark, max(score_event))

    # Section boundary on midi
    midi_sect_bound = get_midi_sect_boundary(pattern, sects, mark,
                                             score_sect_bound, len(event))

    # Renew end time
    t_fin = get_t_fin(max(idx_mapping), ts_cpt)
    ts_cpt.append({"measure": max(idx_mapping) + 1, 't': float(t_fin),
                   "tempo": ts_cpt[-1]["tempo"],
                   "time_signature": ts_cpt[-1]["time_signature"]})

    # Split sections by tempo, time/key signature shifts, for refining phrase boundaries
    key_cpt = get_expanded_key_cpt(mark['key_cpt'], idx_mapping)
    sig_cpt = get_signature_cpt(key_cpt, ts_cpt)
    sub_sect_bound = get_sub_sect_boundary(sig_cpt, midi_sect_bound)

    # Load predicted boundaries
    with open(os.path.join(dirConfig.boundary_dir, f"{composer}-{basename}.pkl"), "rb") as f:
        pred_bound = pickle.load(f)
    seg_bound = pred_bound[n_cluster]

    # Find section boundary based on score section and structural boundary prediction
    intro = None
    if pattern[0] == "I":
        intro_sect = deepcopy(sub_sect_bound["I"])
        intro = [[e['st']['t'], e['ed']['t']] for e in intro_sect]

    # Identify section boundary based on pattern type
    pattern_type, dev_sect = get_pattern_type("".join(pattern))
    if pattern_type == 'A':
        sect_bound = get_struct_A(sub_sect_bound, seg_bound, sig_cpt)
    elif pattern_type == "AB":
        sect_bound = get_struct_A_B(sub_sect_bound, seg_bound, sig_cpt)
    elif pattern_type == "ABA":
        # AB-x-AB or AB-x-AB-y
        sect_bound = get_struct_A_B_A(sub_sect_bound, dev_sect)
    elif pattern_type == "ABC":
        # A-B-A' or A-B-C
        sect_bound = get_struct_A_B_C(sub_sect_bound, seg_bound, sig_cpt)
    else:
        raise ValueError(f"Undefined Pattern Type {pattern}")

    if intro is not None:
        # Todo: consider using ordereddict
        sect_bound = {"intro": intro, **sect_bound}

    # Locate phrases in section
    phrase_bound = update_phrase_bound(pred_bound[n_cluster][0], sect_bound)
    sect_phrase = locate_phrase(phrase_bound, sect_bound, ts_cpt, min_phrase)

    # Get structural label
    struct_labels = get_struct_label(sect_phrase, seg_bound, ts_cpt)
    label_file = os.path.join(dirConfig.output_dir,
                              f"{composer}-{basename}", "human_label1.txt")
    with open(label_file, "w") as f:
        f.write(struct_labels + "\n")

        # res_key = ['tempo', 'time_signature', 'key']
        # sect_event, sect_phrase_pos = {}, {}
        # for name, t_phrases in sect_phrase.items():
        #     sect_phrase_pos[name] = []
        #     phrases = []
        #     for t_phrase in t_phrases:

        #         # Todo: Round position to beat?
        #         st_pos = t_to_bar_pos(t_phrase[0], sig_cpt)
        #         ed_pos = t_to_bar_pos(t_phrase[1], sig_cpt)

        #         phrase = trim_event(event, st_pos, ed_pos)

        #         if not len(phrase):
        #             warnings.warn(f"Empty phrase in {composer/basename}")
        #             continue

        #         sect_phrase_pos[name].append([(st_pos[0], str(st_pos[1])),
        #                                       (ed_pos[0], str(ed_pos[1]))])

        #         i_st, i_ed = min(phrase), max(phrase)
        #         item = {k: phrase[i_st][k] for k in res_key}
        #         item['event'] = [phrase[i]['event'] for i in range(i_st, i_ed + 1)]

        #         phrases += [item]

        #     sect_event[name] = phrases

        # return sect_event, sect_phrase_pos


if __name__ == "__main__":
    import argparse
    from glob import glob

    class dirConfig:
        data_dir = "../../sonata-dataset-phrase"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str,
                        default=dirConfig.data_dir, help="Path to dataset")

    args = parser.parse_args()

    dirConfig.data_dir = args.data_dir
    dirConfig.event_dir = os.path.join(dirConfig.data_dir, "event")
    dirConfig.midi_dir = os.path.join(dirConfig.data_dir, "midi_no_repeat")
    dirConfig.boundary_dir = os.path.join(dirConfig.data_dir, "boundary_predictions")

    dirConfig.output_dir = "../../whole-song-gen-data"

    for composer in ['mozart', 'beethoven', 'scarlatti', 'haydn']:
        file_list = glob(os.path.join(dirConfig.event_dir, composer, "*.json"))

        for event_file in sorted(file_list):
            basename = os.path.basename(event_file).split(".")[0]

            try:
                struct_event, struct_bound = main(composer, basename, dirConfig)
                with open(os.path.join(dirConfig.data_dir, "struct", f"{composer}-{basename}.json"), "w") as f:
                    json.dump(struct_event, f)

                with open(os.path.join(dirConfig.data_dir, "struct_mark", f"{composer}-{basename}.json"), "w") as f:
                    json.dump(struct_bound, f)
            except:
                print(f"Error: {composer}-{basename}")
