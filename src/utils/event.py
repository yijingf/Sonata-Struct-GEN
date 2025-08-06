"""Helper functions to handle event extracted from kern file.
"""

import pretty_midi
import numpy as np
from copy import deepcopy
from fractions import Fraction

from utils.decode import decode_token_to_pm
from utils.common import trim_event, normalize_tp


class Event:
    def __init__(self, token_str):
        self.token_str = token_str

        if self.token_str[0] == "o":
            self.event_type = "onset"
            self.val = Fraction(token_str[2:])

        elif self.token_str[0] == "d":
            self.event_type = "duration"
            self.val = Fraction(token_str[2:])
        else:
            self.event_type = "pitch"
            if "-" in token_str:
                pitch_name = "".join(token_str.split("-"))
                self.val = pretty_midi.note_name_to_number(pitch_name) - 1
            else:
                self.val = pretty_midi.note_name_to_number(token_str)

    def __repr__(self):
        return self.token_str

    def __str__(self):
        return f"{self.event_type}-{self.val}"


def reindex_event(event):
    """Reindex measure from 0.
    """
    new_event = {}
    i_st, i_ed = min(event), max(event)
    for i in range(i_st, i_ed + 1):
        new_event[i - i_st] = deepcopy(event[i])
    return new_event


def no_repeat_pattern(pattern):
    """Reduce the pattern. Omit repetition and jump to the second volta if there are volta brackets at the end of a sub-section.

    Args:
        pattern (list): A list of section name as the music unfolded, extracted from .krn file.

    Returns:
        norep_pattern (list): Reduced pattern.
    """
    norep_pattern = [pattern[0]]
    last_sub_sect_name = pattern[0]

    for sub_sect_name in pattern[1:]:

        if sub_sect_name == last_sub_sect_name:
            continue

        if sub_sect_name[0] != last_sub_sect_name[0]:
            norep_pattern.append(sub_sect_name)

        elif len(sub_sect_name) > len(norep_pattern[-1]):
            norep_pattern.append(sub_sect_name)

        elif len(sub_sect_name) == len(norep_pattern[-1]):
            norep_pattern[-1] = sub_sect_name

        last_sub_sect_name = sub_sect_name

    return norep_pattern


def remove_repeat(pattern):
    """Reduce the pattern. Omit the repetition if there is no volta bracket at the end of a sub-section.

    Args:
        pattern (list): A list of section name as the music unfolded, extracted from .krn file.

    Returns:
        new_pattern (list): Reduced pattern.
    """
    new_pattern = []
    last_sect = ""
    for sect in pattern:
        if sect != last_sect:
            new_pattern.append(sect)
        last_sect = sect
    return new_pattern


def expand_score(score_event, mark, repeat_mode="no_repeat"):
    """Unfold repeats from scores given pattern notation from .krn file.

    Example: 
    Given the pattern ["A", "A1, "A", "A2", "B", "B"], the unfolded scores will be
    1. ["A", "A1", "A", "A2", "B"] when repeat_mode = "volta_only"
    2. ["A", "A1", "A", "A2", "B", "B"] when repeat_mode = "full"
    3. ["A", "A2", "B"] when repeat = "no_repeat

    Args:
        score_event (dict): see example

        ```
            {0:
                {"event": ["o-0", "C4", "d-1"], 
                 "time_signature": "3/4",
                 "tempo": 120, 
                 "key": "C major"}
            }
        ```
        mark (dict): markings on score, e.g.
        ```
            {"pattern": ["A"],
             "A": {"measure": 0, "pos": "1/2"}}}
        ```
        repeat mode (str, optional): `volta_only`, `no_repeat` or `full`. Defaults to `volta_only`.

    Returns:
        event (dict): unfolded score in the same form as "score_event". The measure index always starts from 0.
        idx_mapping (dict): A mapping between measure index of unfolded event and the score event. 
    """

    # Sort sections
    onsets = sorted([(i, v) for i, v in mark["sect"].items()],
                    key=lambda x: (x[1]["measure"], x[1]["pos"]))
    sub_sect_event = get_sub_sect_event(score_event, onsets)

    if repeat_mode == "no_repeat":
        sects = no_repeat_pattern(mark["pattern"])
    elif repeat_mode == "volta_only":
        # Unfold repeats only if there is a volta.
        sects = remove_repeat(mark["pattern"])
    elif repeat_mode == "full":
        sects = mark["pattern"]
    else:
        raise ValueError("Set repeat_mode to 'volta_only', 'no_repeat' or 'full'.")

    event, idx_mapping = concat_event(sub_sect_event, sects, mark["sect"])
    return event, idx_mapping


def flatten_measures(phrase,
                     add_eos=True, eos_token="eos",
                     pad_bar=True, bar_pad_token="sep",
                     max_measure_len=64):
    """Flatten events to token sequences. Pad every measure to "max_measure_len" tokens if "pad_bar=True". Measures longer than max_measure_len are truncated for now.

    Args:
        phrase (list): _description_
        add_eos (bool, optional): Append EOS to the end of a phrase. Defaults to True.
        eos_token (str, optional): EOS token. Defaults to "eos".
        pad_bar (bool, optional): Pad every measure to fixed length. Defaults to True.
        bar_pad_token (str, optional): BAR_PAD token. Defaults to "sep".
        max_measure_len (int, optional): Length of a measure if pad_bar is True. Defaults to 64.

    Returns:
        tokens (list): Flattened token sequence.
    """

    tokens = [phrase["time_signature"], phrase["tempo"]]

    n_measure = len(phrase["note"])

    if pad_bar:
        tokens += phrase["note"][0] + ["bar"]
        measure_len = [0, len(phrase["note"][0]) + 1]

        for i in range(1, n_measure - 1):
            notes = phrase["note"][i]

            pad_len = max_measure_len - len(notes)
            if pad_len > 0:
                notes += [bar_pad_token for _ in range(pad_len)]

            tokens += notes
            tokens += ["bar"]
            measure_len += [len(notes) + 1]

        tokens += phrase["note"][-1]
        tokens += [eos_token]

    else:
        tokens += [token for bar in phrase["note"] for token in bar + ["bar"]]
        tokens[-1] = eos_token

    if not add_eos:
        tokens = tokens[:-1]

    return tokens


def map_measure_to_token(idxs, measure_idxs):
    """Helper function. Map measure indices to token indices.

    Args:
        idxs (numpy.ndarray): Indices of the first onset token in each measure.
        measure_idxs (numpy.ndarray): Indices of measures to be masked.

    Returns:
        mask_idx (list): Index of masked tokens.
    """
    n = len(measure_idxs)
    if not n:
        return []

    if isinstance(measure_idxs, list):
        measure_idxs = np.array(sorted(measure_idxs))

    st_idx = idxs[measure_idxs]
    ed_idx = idxs[measure_idxs + 1]
    mask_idx = [j for i in range(n) for j in list(range(st_idx[i], ed_idx[i]))]

    return mask_idx


def concat_event(sub_sect_event, sects, sect_onset_dict, i_measure=0):
    """Concatenate events from subsections.

    Args:
        sub_sect_event (dict): sub_sect_event returned by get_sub_sect_event.
        sects (list): List of sub-section name.
        sect_onset_dict (dict): A dictionary of section onset, e.g.
            ```
                {"A": {"measure": 0, "pos": "1/2"}}`
            ```
        i_measure (int, optional): Starting measure index of concatenated event. Defaults to 0.

    Returns:
        res (dict): concatenated event
        idx_mapping (dict): index mapping between score event measures and concatenated event measures
    """

    res = {}
    idx_mapping = {}

    for i_sect, sub_sect in enumerate(sects):

        tmp_event = sub_sect_event[sub_sect]
        i_st = min(tmp_event)
        i_ed = max(tmp_event)

        for i in range(i_st, i_ed + 1):
            if i_measure in res:
                res[i_measure]["event"] += deepcopy(tmp_event[i]["event"])
            else:
                # idx_mapping[i_measure] = i
                res[i_measure] = deepcopy(tmp_event[i])

            measure_set = idx_mapping.get(i_measure, set([]))
            measure_set.add(i)
            idx_mapping[i_measure] = measure_set
            i_measure += 1

        if i_sect < len(sects) - 1:
            next_sub_sect = sects[i_sect + 1]
            if sect_onset_dict[next_sub_sect]["pos"] != "0":
                i_measure -= 1

    idx_mapping = {k: list(v) for k, v in idx_mapping.items()}
    return res, idx_mapping


def get_sub_sect_event(event, sub_sect_onset):
    """Segment events by sub-secitons. 

    Args:
        event (dict): Score events. 
        sub_sect_onset (list): A list of sub-section onset, e.g.
            ```
                [("A", {"measure": 0, "pos": "1/2"})]
            ```

    Returns:
        sub_sect_event (dict): events belongs to each sub-section, e.g. 
            ```
                {"A": {0: ["0-0", "C4", "d-1"]}}
            ```
    """
    onsets = deepcopy(sub_sect_onset)
    onsets.append(("Fin", {"measure": max(event) + 1, "pos": "0"}))

    # Get events for each sub section
    sub_sect_event = {}
    for i, v in enumerate(onsets[:-1]):

        sub_sect = v[0]
        start = v[1]["measure"], Fraction(v[1]["pos"])
        end = onsets[i + 1][1]["measure"], Fraction(onsets[i + 1][1]["pos"])
        sub_sect_event[sub_sect] = trim_event(event, start, end)

    return sub_sect_event


def event_to_pm(event, quantize_tp=True):
    """Convert event tokens to pretty_midi.pretty_midi.prettyMIDI

    Args:
        event (dict): measures of token events. e.g. 
            ```
                {0: {"event": ["o-3/2", "C4", "d-1/2"],
                     "time_signature": "2/4",
                     "tempo": 120,
                     "key": "F minor"}}
            ```
        quantize_tp (bool, optional): Quantize tempo to the closest discrete values defined as TEMPO_BIN in utils.constants. Defaults to True.

    Returns:
        pm: pretty_midi.pretty_midi.PrettyMIDI converted from event
        cpt: list of dict, tempo/time signature change point, e.g.
             ```
                [{"measure": 0, "t": 0, "tempo": 120, "time_signature": "2/4"}]
             ```
    """
    i_st, i_ed = min(event), max(event)

    cpt = []  # time/key signature change points
    tokens = []
    prev_tp, prev_ts = None, None

    t_offset, next_t_offset = 0, 0
    inst = pretty_midi.Instrument(program=0)

    # Convert measures of event to midi
    for i in range(i_st, i_ed + 1):

        if quantize_tp:
            # Quantize tempo to discrete values
            tp = normalize_tp(event[i]["tempo"])
        else:
            # Todo: Original one, will be removed in the future
            tp = int(event[i]["tempo"] / 12) * 12  # To avoid weird ticks
            # ts = normalize_ts(event[i]["time_signature"])

        event[i]["tempo"] = tp
        ts = event[i]["time_signature"]

        if tp != prev_tp or ts != prev_ts:

            # record tempo/time signature change point
            cpt.append({"measure": i, "t": next_t_offset,
                        "tempo": tp, "time_signature": ts})
            prev_tp, prev_ts = tp, ts

            if len(tokens):
                measure_pm = decode_token_to_pm(tokens, t_offset=t_offset)
                inst.notes += measure_pm.instruments[0].notes

            tokens = [f"ts-{ts}", f"tp-{tp}"]
            t_offset = next_t_offset

        tokens += event[i]["event"] + ["bar"]
        if len(ts.split("/")) > 2:
            ts_frac = Fraction(ts[:-2]) / 4
        else:
            ts_frac = Fraction(ts)
        t_measure = float(ts_frac * 4) * 60 / tp
        next_t_offset += t_measure

    if len(tokens):
        measure_pm = decode_token_to_pm(tokens, t_offset=t_offset)
        inst.notes += measure_pm.instruments[0].notes

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)

    return pm, cpt


def trunc_duration(event):
    for i in event:
        ts = Fraction(event[i]['time_signature']) * 4
        for j, token in enumerate(event[i]['event']):
            e = Event(token)
            if e.event_type == 'duration' and e.val > ts:
                event[i]['event'][j] = f"d-{ts}"

    return event
