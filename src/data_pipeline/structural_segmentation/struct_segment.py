"""Structural Segmentation

Returns:
    section-phrase structrure
"""
# The end of last segment/section is always the end of a complete measure.
# Todo: 1. Use OrderedDict instead of list!

import os
import re
import json
import pickle
import pretty_midi
import numpy as np
from copy import deepcopy
from fractions import Fraction
from collections import OrderedDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.common import load_event, get_file_list
from utils.event import expand_score, no_repeat_pattern
from utils.align import get_t_bar, get_t_fin, t_to_bar_pos, bar_pos_to_t
from structural_segmentation.A_sect_bound import get_struct_A
from structural_segmentation.struct_label import get_struct_label
from structural_segmentation.longest_repeating_pattern import find_lrp

# Constants
from utils.constants import DATA_DIR
event_dir = os.path.join(DATA_DIR, "event")
midi_dir = os.path.join(DATA_DIR, "midi")
boundary_dir = os.path.join(DATA_DIR, "boundary_predictions")


class ScoreInfo:

    def __init__(self, composer, basename):
        self.composer = composer
        self.basename = basename

        # Load event file
        event_file = os.path.join(event_dir, self.composer, f"{self.basename}.json")
        self.score_event, self.mark = load_event(event_file)
        self.event, self.idx_mapping = expand_score(self.score_event, self.mark, "no_repeat")

        self.sects = self.get_score_section()
        self.pattern = self.get_pattern()
        return

    def load_midi(self, melody=False):
        if melody:
            midi_dir = os.path.join(DATA_DIR, "melody_midi")
        else:
            midi_dir = os.path.join(DATA_DIR, "midi")
        midi_file = os.path.join(midi_dir, self.composer, f"{self.basename}.mid")
        pm = pretty_midi.PrettyMIDI(midi_file)
        return pm

    def get_pattern(self):
        pattern = []
        for i, sect in enumerate(self.sects):
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

    def get_score_section(self):
        """Merge score sub-section into sections

        Args:
            mark (_type_): _description_

        Returns:
            _type_: _description_
        """
        sub_sects = no_repeat_pattern(self.mark["pattern"])
        # Merge volta into section, e.g. A1/A2 -> A
        sects = merge_sub_sect_name(sub_sects)

        # Merge short intro (<4 measure) into adjacent sub-section
        if self.mark["pattern"][0] == "I":
            len_I = self.mark["sect"]['A']["measure"] - self.mark["sect"]["I"]["measure"]
            if len_I < 4:
                sects = sects[1:]
                sects[0] = ["I"] + sects[0]

        if "II" in self.mark["pattern"]:
            II_idx = sects.index(["II"])
            next_sect = sects[II_idx + 1][0]
            len_II = self.mark["sect"][next_sect]["measure"] - self.mark["sect"]["II"]["measure"]
            if len_II < 4:
                sects = sects[:II_idx] + sects[II_idx + 1:]
                sects[II_idx] = ["II"] + sects[II_idx]

        if len(sects) > 1 and sects[0] == ['A']:
            next_sect = sects[1][0]
            if self.mark["sect"][next_sect]["measure"] - self.mark["sect"]['A']["measure"] < 4:
                sects = sects[1:]
                sects[0] = ['A'] + sects[0]
        return sects

    def get_score_sub_sect_boundary(self):
        n_measure = len(self.score_event)
        sub_sect_onset = sorted([(i, v) for i, v in self.mark["sect"].items()],
                                key=lambda x: (x[1]["measure"], x[1]["pos"]))
        sub_sect_onset.append(("Fin", {"measure": n_measure + 1, "pos": "0"}))

        sub_sect_bound = {}
        for i, (sub_sect_name, onset) in enumerate(sub_sect_onset[:-1]):
            sub_sect_bound[sub_sect_name] = {'st': onset,
                                             'ed': sub_sect_onset[i + 1][-1]}
        return sub_sect_bound

    def get_midi_sect_boundary(self):
        """Converting score sections to midi sections
        """
        # Sub-section boundary on score
        self.score_sect_bound = self.get_score_sub_sect_boundary()

        # Section boundary on midi
        i_measure = 0
        midi_sect_onset = {}
        is_first_repeat = False

        # Unfold pattern
        for i_sect, sect_name in enumerate(self.pattern):

            sub_sect_name = self.sects[i_sect][0]

            if sect_name in midi_sect_onset:
                sect_name = f"{sect_name}-repeat"
                is_first_repeat = not (is_first_repeat)

            pos = Fraction(self.mark["sect"][sub_sect_name]["pos"])
            if pos != 0 and is_first_repeat:
                i_measure -= 1
            midi_sect_onset[sect_name] = {"measure": i_measure, "pos": pos}

            for sub_sect_name in self.sects[i_sect]:
                bound = self.score_sub_sect_bound[sub_sect_name]
                i_measure += bound['ed']["measure"] - bound['st']["measure"]

        midi_sect_onset = sorted([(i, v) for i, v in midi_sect_onset.items()],
                                 key=lambda x: (x[1]["measure"], x[1]["pos"]))
        midi_sect_onset.append(("Fin", {"measure": len(self.event), "pos": 0}))

        self.midi_sect_bound = OrderedDict()
        for i, (sect_name, onset) in enumerate(midi_sect_onset[:-1]):
            self.midi_sect_bound[sect_name] = {'st': onset,
                                               'ed': midi_sect_onset[i + 1][-1]}
        return self.midi_sect_bound

    def load_cpt(self):
        # Load tempo/time signature change points
        map_file = os.path.join(midi_dir, self.composer, f"{self.basename}.json")
        with open(map_file) as f:
            self.ts_cpt = json.load(f)["cpt"]

        # Renew end time
        self.t_fin = get_t_fin(max(self.event), self.ts_cpt)
        self.ts_cpt.append({"measure": max(self.idx_mapping) + 1, 't': float(self.t_fin),
                            "tempo": self.ts_cpt[-1]["tempo"],
                            "time_signature": self.ts_cpt[-1]["time_signature"]})
        return

    def get_expanded_key_cpt(self):
        expanded_key_cpt = []

        score_key_cpt_idx = {v['measure']: i for i, v in enumerate(self.mark['key_cpt'])}
        reversed_idx_mapping = get_reverse_idx_mapping(self.idx_mapping)

        for idx_score, idx_midi in reversed_idx_mapping:
            i = score_key_cpt_idx.get(idx_score, None)
            if i is not None:
                expanded_key_cpt.append({"measure": int(idx_midi),
                                         "pos": self.mark['key_cpt'][i]['pos'],
                                         "key": self.mark['key_cpt'][i]['key']})
        return expanded_key_cpt

    def get_signature_cpt(self):
        """
        Tempo, time signature, key signature changes
        """
        key_cpt = self.get_expanded_key_cpt()
        for entry in self.ts_cpt[:-1]:
            entry['pos'] = Fraction(0)
            entry['key'] = self.event[entry['measure']]['key']

        self.sig_cpt = []
        last_measure = self.ts_cpt[0]['measure'] - 1

        for e in sorted(self.ts_cpt + key_cpt, key=lambda x: x['measure']):
            if e['measure'] == last_measure:
                continue

            last_measure = e['measure']

            if 't' in e:
                if 'pos' not in e:
                    e['pos'] = 0
                self.sig_cpt.append(e)
                continue

            t_bar = get_t_bar(**self.sig_cpt[-1])
            t = (e['measure'] - self.sig_cpt[-1]['measure']) * t_bar
            t -= self.sig_cpt[-1]['pos'] * 60 / self.sig_cpt[-1]['tempo']
            e['t'] = float(self.sig_cpt[-1]['t']) + float(t)
            if 'pos' in e:
                e['pos'] = Fraction(e['pos'])
                e['t'] += float(e['pos'] * 60 / self.sig_cpt[-1]['tempo'])
            else:
                e['pos'] = 0

            for key in ['tempo', 'time_signature']:
                e[key] = deepcopy(self.sig_cpt[-1][key])

            self.sig_cpt.append(e)

        return

    def get_sub_sect_boundary(self):

        self.get_signature_cpt()

        # Revise the last score section boundary
        pos = Fraction(self.sig_cpt[-1]["time_signature"]) * 4
        last_sect = list(self.midi_sect_bound.keys())[-1]
        n_measure = self.midi_sect_bound[last_sect]['ed']["measure"] - 1
        self.midi_sect_bound[last_sect]['ed'] = {"measure": n_measure, "pos": pos}

        # Combine repeat signs with time/key signature and tempo shifts
        bounds = [(i['st']["measure"], i['st']["pos"]) for i in self.midi_sect_bound.values()]
        bounds += [(i["measure"], i['pos']) for i in self.sig_cpt[:-1]]
        bounds = sorted(set(bounds))

        sub_sect_bound = OrderedDict()
        for name, e in self.midi_sect_bound.items():
            st = (e['st']["measure"], e['st']["pos"])
            ed = (e['ed']["measure"], e['ed']["pos"])
            seg = [st] + [i for i in bounds if i < ed and i > st] + [ed]
            seg = [(i[0], i[1], float(bar_pos_to_t(i, self.sig_cpt))) for i in seg]
            sub_sect_bound[name] = []
            for i, pos in enumerate(seg[:-1]):
                sub_sect_bound[name] += [{'st': {"measure": pos[0], "pos": pos[1], 't': pos[2]},
                                          'ed': {"measure": seg[i + 1][0], "pos": seg[i + 1][1],
                                                 't': seg[i + 1][2]}}]

        # post process for A-x-A
        if len(sub_sect_bound[last_sect]) > 1:
            fin_entry = deepcopy(sub_sect_bound[last_sect][-1]['ed'])
            if fin_entry['measure'] == sub_sect_bound[last_sect][-2]['ed']['measure']:
                sub_sect_bound[last_sect] = sub_sect_bound[last_sect][:-1]
                sub_sect_bound[last_sect][-1]['ed'] = fin_entry

        return sub_sect_bound

    def load_pred_bound(self):
        # Load predicted boundaries
        with open(os.path.join(boundary_dir, f"{self.composer}-{self.basename}.pkl"), "rb") as f:
            self.pred_bound = pickle.load(f)
        return


def verify_sect_bound(sect_bound, ts_cpt, n_bar=4):
    t_shift = np.array([i['t'] for i in ts_cpt])
    for name in sect_bound:
        for i, e in enumerate(sect_bound[name]):
            idx = np.where(t_shift <= 0)[0][-1]
            t_bar = float(get_t_bar(**ts_cpt[idx]))
            if e[1] - e[0] < n_bar * t_bar:
                return False
    return True


def post_process_sect_bound(sect_bound, ts_cpt, n_bar=4):
    t_shift = np.array([i['t'] for i in ts_cpt])
    for name in sect_bound:
        t_sub_sect, i_sub_sect = [], []
        for i, e in enumerate(sect_bound[name]):
            idx = np.where(t_shift <= 0)[0][-1]
            t_bar = float(get_t_bar(**ts_cpt[idx]))
            t_sub_sect += [t_bar]
            if e[1] - e[0] < n_bar * t_bar:
                i_sub_sect += [i]

        if not len(i_sub_sect):
            continue

        new_interval = []
        p = 0
        for i in i_sub_sect:
            if i < len(sect_bound[name]) - 1:
                new_interval += deepcopy(sect_bound[name][p:i])
                if t_sub_sect[i] == t_sub_sect[i + 1]:
                    new_interval += [[deepcopy(sect_bound[name][i][0]),
                                     deepcopy(sect_bound[name][i + 1][-1])]]
                else:
                    new_interval += deepcopy(sect_bound[name][i:i + 2])
                p = i + 2
            else:
                new_interval += deepcopy(sect_bound[name][p:i - 1])
                if t_sub_sect[i] == t_sub_sect[i - 1]:
                    new_interval += [[deepcopy(sect_bound[name][i - 1][0]),
                                      deepcopy(sect_bound[name][i][-1])]]
                else:
                    new_interval += deepcopy(sect_bound[name][i - 1:])
                p = i + 1

        new_interval += sect_bound[name][p:]
        sect_bound[name] = new_interval
    return sect_bound


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


def augment_t_recap(t_recap, sig_cpt, sub_sect_bound, beat_pt, align_pos=True):
    # Try round to closest section boundary
    augmented_t = round_to_sect_bound(t_recap, sub_sect_bound)
    if augmented_t:
        t_recap = augmented_t
    else:
        if align_pos:
            # Make sure recap starts at the same beat position as expo
            measure, beat = t_to_bar_pos(t_recap, sig_cpt)
            # print("raw_t_recap", bar_pos_to_t((measure, beat), sig_cpt))
            measure += get_delta_measure(beat, beat_pt)
            t_recap = bar_pos_to_t((measure, beat_pt), sig_cpt)
        else:
            measure, beat = t_to_bar_pos(t_recap, sig_cpt)
            t_recap = bar_pos_to_t((measure, beat_pt), sig_cpt)
    return t_recap


def get_t_recap(t_candidates, sig_cpt, sub_sect_bound, beat_pt,
                thresh=0.45, n_bound=3, align_pos=True):

    t_fin = sig_cpt[-1]['t']
    if len(t_candidates):

        i_candidate = 0
        while i_candidate < len(t_candidates):

            # Try round to closest section boundary
            raw_t_recap = t_candidates[i_candidate]
            t_recap = augment_t_recap(raw_t_recap, sig_cpt, sub_sect_bound,
                                      beat_pt, align_pos)

            # make sure t_recap is in the middle of a piece(t_recap/f_fin)
            if len(t_candidates) - i_candidate <= n_bound and t_recap / t_fin > thresh:
                break

            t_recap = t_fin
            i_candidate += 1
    else:
        # Not ABA'
        t_recap = t_fin
    return t_recap


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


def round_to_sect_bound(t_pred, sub_sect_bound, t_thresh=5):
    """Round predicted boundary time to its closest tempo, key/time signature shift time.
    Return [rounded t_pred, True] if a boundary time is found in t_pred +/- t_thresh
    else return [t_pred, False].

    Args:
        t_pred (float): predicted boundary time
        sig_cpt (array, optional): _description_. Defaults to None.
        t_thresh (int, optional): _description_. Defaults to 5.
    """
    shift_t = []
    for name in sub_sect_bound:
        shift_t += [i['ed']['t'] for i in sub_sect_bound[name]]

    t_diff = np.abs(t_pred - shift_t)
    idx = np.argmin(t_diff)
    if t_diff[idx] <= t_thresh:
        return shift_t[idx]
    return None


def get_reverse_idx_mapping(idx_mapping):
    reversed_idx_mapping = []
    for midi_idx, score_idxs in idx_mapping.items():
        reversed_idx_mapping += [(v, midi_idx) for v in score_idxs]
    return reversed_idx_mapping


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
    t_bounds, labels = boundaries
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
    i_pt = np.argmin(np.abs(t_bounds[:, 0] - t_A))
    t_pt = t_bounds[labels == labels[i_pt]][:, 0]
    t_fin = sig_cpt[-1]['t']

    # Make sure development has at least 4 measures.
    for i, e in enumerate(sig_cpt):
        if e['t'] > t_dev:
            break
    t_dev_min = get_t_bar(**sig_cpt[i - 1]) * 4
    t_candidates = t_pt[(t_pt > t_dev + t_dev_min)]
    t_recap = get_t_recap(t_candidates, sig_cpt, sub_sect_bound, beat_pt)

    # If identified t_recap is not in the middle of the piece.
    if t_recap / sig_cpt[-1]['t'] > 0.75:
        t_recap = sig_cpt[-1]['t']

    if t_recap == sig_cpt[-1]['t']:
        # check if other material in primary theme is repeated
        t_pt = t_bounds[labels == labels[i_pt + 1]][:, 0]
        # t_pt = t_bounds[np.where(labels == labels[i_pt + 1])[0] - 1][:, 0]
        t_candidates = t_pt[(t_pt > t_dev + t_dev_min) & (t_pt / t_fin < 0.8)]
        if not len(t_candidates):
            t_pt = t_bounds[np.where(labels == labels[i_pt + 2])[0] - 2][:, 0]
            t_candidates = t_pt[(t_pt > t_dev + t_dev_min) & (t_pt / t_fin < 0.8)]
        t_recap = get_t_recap(t_candidates, sig_cpt, sub_sect_bound, beat_pt,
                              align_pos=False)

    # if no valid recapitulation identified
    if t_recap == sig_cpt[-1]['t'] and len(sub_sect_bound['B']) > 1:
        # check if there is any mark in section B is potentially a boundary
        t_recap = sub_sect_bound['B'][1]['st']['t']

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

    t_bounds, labels = boundaries
    labels = np.array(labels)

    # Exposition
    # Assume exposure always exists in A
    sub_sect_names = list(sub_sect_bound.keys())
    i_expo = sub_sect_names.index('A')
    t_A, _, beat_pt = get_t_primary_theme(sub_sect_bound)

    # Find other occurences of primary theme
    t_fin = sig_cpt[-1]['t']
    i_pt = np.argmin(np.abs(t_bounds[:, 0] - t_A))
    t_pt = t_bounds[labels == labels[i_pt]][:, 0]
    t_pt = t_pt[t_pt > t_bounds[:, 0][i_pt]]

    is_invalid_recap = (len(t_pt) == 1 and t_pt[0] / t_fin < 0.3)
    if not len(t_pt) or is_invalid_recap:
        t_pt = t_bounds[labels == labels[i_pt + 1]][:, 0]
        t_pt = t_pt[t_pt > t_bounds[:, 0][i_pt + 1]]

    # Recapitulation
    i_B = sub_sect_names.index('B')
    t_B = sub_sect_bound['B'][0]['st']['t']
    n_sub_sect = len(sub_sect_bound)

    # Make sure development has at least 4 measures.
    for i, e in enumerate(sig_cpt):
        if e['t'] > t_B:
            break
    t_B_min = get_t_bar(**sig_cpt[i - 1]) * 4

    # heuristic: t_recap should be in the middle of a mov.
    t_candidates = t_pt[(t_pt > t_B + t_B_min) &
                        (t_pt < 0.8 * t_fin) & (t_pt > 0.4 * t_fin)]

    if not len(t_candidates) and len(t_pt) == 1 and t_pt[0] / t_fin > 0.3 and t_pt[0] / t_fin < 0.9:
        t_recap = augment_t_recap(t_pt[0], sig_cpt, sub_sect_bound, beat_pt)
    elif len(t_candidates) == 1:
        t_recap = augment_t_recap(t_candidates[0], sig_cpt, sub_sect_bound, beat_pt)
    else:
        t_recap = get_t_recap(t_candidates, sig_cpt, sub_sect_bound, beat_pt,
                              thresh=0.5, n_bound=4)

    i_recap = n_sub_sect - 1
    for i in range(i_B, n_sub_sect):
        name = sub_sect_names[i]
        if t_recap < sub_sect_bound[name][-1]['ed']['t']:
            i_recap = i
            break

    # Development
    if t_recap == sig_cpt[-1]['t']:
        i_dev = i_B
        t_dev = sub_sect_bound['B'][0]['st']['t']
    else:
        # If there are multiple entries of primary theme
        i_dev = None
        t_dev = None
        if len(t_pt) > 2:

            # Dev is likely to be the material after the second entry of primary theme
            idx = np.argmin(np.abs(t_pt - t_recap))
            t_second_entry = t_pt[idx - 1]

            flag = False
            for name in sub_sect_names[:i_recap + 1]:
                if flag:
                    break
                bounds = sub_sect_bound[name]
                if t_second_entry >= bounds[-1]['ed']['t']:
                    continue
                else:
                    for j, bound in enumerate(bounds):
                        if bound['ed']['t'] >= t_recap:
                            flag = True
                            break

                        if t_second_entry < bound['ed']['t']:
                            t_dev = bound['ed']['t']
                            i_dev = sub_sect_names.index(name)
                            flag = True
                            break

        # if no valid t_dev identified
        if i_dev is None or t_dev / t_fin < 0.15:
            i_dev = get_i_dev_A_B_C(sub_sect_bound, t_recap, i_expo, i_recap) - 1
            t_dev = sub_sect_bound[sub_sect_names[i_dev]][-1]['ed']['t']

    # Fill score subsection boundary into section
    sect_bound = {'expose': [], 'dev': []}

    # Exposition
    for name in list(sub_sect_bound)[i_expo: i_dev + 1]:
        for j, e in enumerate(sub_sect_bound[name]):
            if t_dev >= e['ed']['t']:
                sect_bound['expose'] += [[e['st']['t'], e['ed']['t']]]
                if t_dev == e['ed']['t']:
                    break
            else:
                j -= 1
                break

    # Development
    for e in sub_sect_bound[sub_sect_names[i_dev]][j + 1:]:
        if e['ed']['t'] > t_recap:
            if e['st']['t'] == t_recap:
                break
            sect_bound['dev'] += [[e['st']['t'], t_recap]]
            break
        else:
            sect_bound['dev'] += [[e['st']['t'], e['ed']['t']]]

    for name in list(sub_sect_bound)[i_dev + 1: i_recap + 1]:
        for j, e in enumerate(sub_sect_bound[name]):
            if e['ed']['t'] > t_recap:
                if e['st']['t'] == t_recap:
                    break
                sect_bound['dev'] += [[e['st']['t'], t_recap]]
                break
            else:
                sect_bound['dev'] += [[e['st']['t'], e['ed']['t']]]

    if t_recap == sig_cpt[-1]['t']:
        # recapitulation not found
        return sect_bound

    # Recapitulation
    if e['ed']['t'] > t_recap:
        sect_bound['recap'] = [[t_recap, e['ed']['t']]]
    # elif e['ed']['t'] == t_recap:
    #     e = sub_sect_bound[sub_sect_names[i_recap]][i]
    #     sect_bound['recap'] = [[t_recap, e['ed']['t']]]

    for e in sub_sect_bound[sub_sect_names[i_recap]][j + 1:]:
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


def select_pred_bound(pred_bound, ts_cpt, n_min=2, n_max=10):
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
    labels = pred_bound[k]
    return t_bounds, labels


def get_air_struct(score_info):
    score_info.get_midi_sect_boundary()
    bounds = score_info.midi_sect_bound
    if len(bounds) <= 1:
        print(f"No section boundary in {score_info.event_file}.")
        return

    score_info.load_cpt()

    pos_bounds = []
    for bound in bounds.values():
        measure = bound['st']['measure']
        pos = bound['st']['pos']
        t = bar_pos_to_t((measure, pos), score_info.ts_cpt)
        pos_bounds += [{"end pos": (measure, str(pos)), "end t": float(t)}]

    pos = Fraction(score_info.ts_cpt[-1]['time_signature']) * 4
    pos_bounds = pos_bounds[1:] + [{"end pos": (max(score_info.event), str(pos)),
                                    "end t": float(score_info.t_fin)}]
    struct = {"part i": pos_bounds}
    return struct


def mozart_sonata06_3l(score_info):
    score_info.load_cpt()

    pm = score_info.load_midi()

    notes = sorted(pm.instruments[0].notes, key=lambda x: x.start)
    st_dur = np.array([(note.start, note.duration) for note in notes])
    t_bounds = sorted(
        set([0] + list(np.sum(st_dur[st_dur[:, 1] >= 2], axis=1)) + [score_info.t_fin]))

    true_bounds = [t_bounds[-1]]
    thresh = 4 * 60 / 60 * 4
    for t in t_bounds[::-1]:
        if true_bounds[-1] - t > thresh:
            true_bounds.append(t)

    pos_bounds = []
    for t in true_bounds[::-1]:
        measure, pos = t_to_bar_pos(t, score_info.ts_cpt)
        pos_bounds += [{"end pos": (measure, str(pos)), "end t": float(t)}]

    pos = Fraction(score_info.ts_cpt[-1]['time_signature']) * 4
    pos_bounds[-1]['end pos'] = (max(score_info.event), str(pos))
    struct = {"part i": pos_bounds[1:]}
    return struct


def mozart_sonata06_3m(score_info):

    n_measure_phrase = 8
    patterns = [[81, 73], [69, 73]]

    pm = score_info.load_midi(melody=True)

    score_info.load_cpt()
    tp = score_info.ts_cpt[0]['tempo']
    ts = score_info.ts_cpt[0]['time_signature']
    t_thresh = 60 / tp * Fraction(ts) * 4 * n_measure_phrase

    notes = sorted(pm.instruments[0].notes, key=lambda x: x.start)
    st_pitch = np.array([(note.start, note.pitch) for note in notes])

    pitch_pair = [[v, st_pitch[i + 1, 1]] for i, v in enumerate(st_pitch[:-1, 1])]

    pos_bounds = []
    t_bounds = [1]
    for i, pair in enumerate(pitch_pair):
        t = st_pitch[i][0]
        if pair in patterns:
            if t - t_bounds[-1] < t_thresh:
                continue
            else:
                t_bounds.append(t)
                measure, pos = t_to_bar_pos(t, score_info.ts_cpt)
                pos_bounds += [{"end pos": (measure, str(pos)), "end t": float(t)}]

    pos = Fraction(score_info.ts_cpt[-1]['time_signature']) * 4
    pos_bounds += [{"end pos": (max(score_info.event), str(pos)),
                    "end t": float(score_info.t_fin)}]
    struct = {"part i": pos_bounds}
    return struct


def get_sect_phrase_struct(score_info, min_phrase=4):
    if bool(re.search(r'[0-9]+[a-z].json', score_info.basename)):
        print(f"{score_info.composer}-{score_info.basename} is potentially an air with variations.")
        if score_info.composer == "mozart" and score_info.basename == "sonata06-3l":
            return mozart_sonata06_3l(score_info)
        elif score_info.composer == "mozart" and score_info.basename == "sonata06-3m":
            return mozart_sonata06_3m(score_info)
        return get_air_struct(score_info)

    score_info.get_midi_sect_boundary()

    # Split sections by tempo, time/key signature shifts, for refining phrase boundaries
    score_info.load_cpt()
    sub_sect_bound = score_info.get_sub_sect_boundary()

    # Find section boundary based on score section and structural boundary prediction
    intro = None
    if score_info.pattern[0] == "I":
        intro_sect = deepcopy(sub_sect_bound["I"])
        intro = [[e['st']['t'], e['ed']['t']] for e in intro_sect]

    score_info.load_pred_bound()

    # Identify section boundary based on pattern type
    pattern_type, dev_sect = get_pattern_type("".join(score_info.pattern))
    if pattern_type == 'A':
        sect_bound = get_struct_A(score_info.pred_bound, score_info.sig_cpt)
    elif pattern_type == "AB":
        sect_bound = get_struct_A_B(sub_sect_bound, score_info.pred_bound[-1], score_info.sig_cpt)
    elif pattern_type == "ABA":
        # AB-x-AB or AB-x-AB-y
        sect_bound = get_struct_A_B_A(sub_sect_bound, dev_sect)
    elif pattern_type == "ABC":
        # A-B-A' or A-B-C
        sect_bound = get_struct_A_B_C(sub_sect_bound, score_info.pred_bound[-1], score_info.sig_cpt)
    else:
        raise ValueError(f"Undefined Pattern Type {score_info.pattern}")

    if intro is not None:
        # Todo: consider using ordereddict
        sect_bound = {"intro": intro, **sect_bound}

    # Merge small sub-section boundary
    if not verify_sect_bound(sect_bound, score_info.ts_cpt):
        sect_bound = post_process_sect_bound(sect_bound, score_info.ts_cpt)

    # Locate phrases in section
    t_bound, _ = select_pred_bound(score_info.pred_bound, score_info.ts_cpt)
    t_bound = np.array([t_bound[:-1], t_bound[1:]]).T
    phrase_bound = update_phrase_bound(t_bound, sect_bound)
    # phrase_bound = update_phrase_bound(pred_bound[n_cluster][0], sect_bound)
    sect_phrase = locate_phrase(phrase_bound, sect_bound, score_info.ts_cpt, min_phrase)

    res = {}
    for sect, phrases in sect_phrase.items():
        res[sect] = []
        for phrase in phrases:
            pos = t_to_bar_pos(phrase[1], score_info.sig_cpt, scale=2)
            t = bar_pos_to_t(pos, score_info.sig_cpt)
            res[sect] += [{'end pos': [pos[0], str(pos[1])], 'end t': float(t)}]

    res[sect][-1]['end pos'] = [max(score_info.event) + 1, "0"]
    res[sect][-1]['end t'] = float(score_info.t_fin)

    return res


def main():
    event_file_list = sorted(get_file_list(event_dir))
    for event_file in event_file_list:
        try:
            composer, basename = (event_file.split('.json')[0]).split('/')[-2:]
            score_info = ScoreInfo(composer, basename)
            struct = get_sect_phrase_struct(score_info)
            output_file = os.path.join(DATA_DIR, "struct_mark", f"{composer}-{basename}.json")
            with open(output_file, "w") as f:
                json.dump(struct, f)
        except:
            print(f"Error: {composer}-{basename}")
    return


def main_diffusion(output_dir):
    # Get structural labels for diffusion model.
    event_file_list = sorted(get_file_list(event_dir))
    for event_file in event_file_list:
        try:
            composer, basename = (event_file.split('.json')[0]).split('/')[-2:]
            score_info = ScoreInfo(composer, basename)
            struct_file = os.path.join(DATA_DIR, "struct_mark", f"{composer}-{basename}.json")
            if os.path.exists(struct_file):
                with open(struct_file) as f:
                    struct = json.load(f)
            else:
                struct = get_sect_phrase_struct(score_info)

            score_info.load_cpt()
            score_info.load_pred_bound()
            t_bound, labels = select_pred_bound(score_info.pred_bound, score_info.ts_cpt)
            t_bound = np.array([t_bound[:-1], t_bound[1:]]).T

            struct_label = get_struct_label(struct, (t_bound, labels), score_info.ts_cpt)
            label_file = os.path.join(output_dir, f"{composer}-{basename}", "human_label1.txt")
            with open(label_file, "w") as f:
                f.write(struct_label + "\n")

        except:
            print(f"Error: {composer}-{basename}")
    return


if __name__ == "__main__":
    main()
