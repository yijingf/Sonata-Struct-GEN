"""Helper functions
"""

import json
import numpy as np
from copy import deepcopy
from music21 import pitch
from fractions import Fraction

# Constants
from utils.constants import PITCH_OFFSET_DICT, TEMPO_BIN


def token2v(token):
    """Convert onset/duration token to value.
    Args:
        token (str): _description_

    Returns:
        Fraction: Value as fraction.
    """
    return Fraction(token.split("-")[-1])


def ts_tp_ratio(ts, tp):
    ratio = 1
    ts_denom = int(ts.split("/")[-1])
    if tp <= 72 and ts_denom == 8:
        ratio = Fraction(2, 1)

    elif tp >= 192 and ts_denom == 2:
        ratio = Fraction(1, 2)

    return ratio


def normalize_tp(tp):
    """Normalize tempo to its closest regular tempo.

    Args:
        tp (int): original tempo

    Returns:
        int: normalized tempo
    """
    idx = np.argmin(np.abs(tp - TEMPO_BIN))
    return int(TEMPO_BIN[idx])


def normalize_ts(ts, base=4):
    """Normalize time signature with denominator of 4.

    Args:
        ts (Fraction): Time signature as fraction.
        base (int, optional): Denominator of the time signature. Defaults to 4.

    Returns:
        str: normalized time signature string with a denominator of 4
    """
    if isinstance(ts, str):
        ts = Fraction(ts)
    ts_num = ts.numerator * Fraction(base, ts.denominator)
    normed_ts = f"{ts_num}/{base}"
    return normed_ts


def normalize_ts_tp(ts, tp):
    """Normalize time signature and tempo. See `normalize_ts` and `normalize_tp` for more details.

    Args:
        ts (Fraction): Time signature as fraction.
        tp (int): tempo.

    Returns:
        str, int : normalized time signature, normalized tempo
    """
    ratio = ts_tp_ratio(ts, tp)
    ts = Fraction(ts) * ratio
    normed_ts = normalize_ts(ts)
    normed_tp = normalize_tp(tp * ratio)
    return normed_ts, normed_tp


def pitch_transpose(pitch_token, offset):
    note_ps = pitch.Pitch(pitch_token).ps + offset
    return pitch.Pitch(note_ps).nameWithOctave


def time_transpose(token, ratio=1):

    token_type, v = token.split("-")
    t = Fraction(v) * ratio

    return f"{token_type}-{t}"


def normalize_pitch(tokens, key=None):

    pitch_offset = PITCH_OFFSET_DICT[key]

    if not pitch_offset:
        return tokens

    new_tokens = []
    for token in tokens:
        if token[0] in ["o", "d"]:
            new_tokens.append(token)
        else:
            new_tokens.append(pitch_transpose(token, pitch_offset))

    return new_tokens


def normalize_event(event, adjust_ts_tp=True):
    """Transpose all pitches to C major/minor; normalize note onset/duration in pieces with irregular time signature and tempo. NOTE: This operation is performed in place!

    Args:
        event (_type_): _description_
    """

    i_st = min(event)
    pitch_offset = PITCH_OFFSET_DICT[event[i_st]["key"].split()[0]]

    for measure in event.values():
        ratio = ts_tp_ratio(measure["time_signature"], measure["tempo"])

        for i, token in enumerate(measure["event"]):
            if token[0] in ["o", "d"]:
                if ratio != 1 and adjust_ts_tp:
                    measure["event"][i] = time_transpose(token, ratio)
            else:
                if pitch_offset:
                    measure["event"][i] = pitch_transpose(token, pitch_offset)

    return event


def load_note_event(fname=None, event=None):
    """Read note event and structure notation from preprocessed event file.

    Args:
        fname (str): File name.

    Returns:
        note_event(dict): note event tokens, key signature, time signature and tempo separated by measure.
    """
    if not event:
        with open(fname) as f:
            event = json.load(f)

    note_event = {}
    for i in event:
        note_event[int(i)] = event[i].copy()
    return note_event


def load_event(fname):
    """Read note event and structure notation from preprocessed event file. The preprocessed event file should have at least two keys: "note" and "struct".

    Args:
        fname (str): File name.

    Returns:
        note_event (dict): note event tokens, key signature, time signature and tempo separated by measure, e.g.
        ```
            {0:
                {"event": ["o-0", "C4", "d-1"], 
                 "time_signature": "3/4",
                 "tempo": 120, 
                 "key": "C major"}
            }
        ```
        struct (dict): structure notation on score, e.g.
            ```
                {"pattern": ["A"],
                 "A": {"idx": 0, "onset": "o-0"}}}
            ```
    """
    with open(fname) as f:
        event = json.load(f)

    note_event = {}
    for i in event["note"]:
        note_event[int(i)] = event["note"][i].copy()

    struct = event["struct"]

    return note_event, struct


def trim_event(measures, start=(0, 0), end=(0, 0)):
    """Trim a list of measures given the start/end time.

    Args:
        event (list): _description_
        start (tuple, optional): (measure index, beat/quarter note within a measure). Defaults to (0, 0).
        end (tuple, optional): _description_. Defaults to (0, 0).

    Returns:
        dict: _description_
    """
    i_st, offset_st = start
    i_ed, offset_ed = end

    seg_measures = {}
    for i_measure in range(i_st, i_ed):
        seg_measures[i_measure] = deepcopy(measures[i_measure])

    # Add notes from last measure
    flag = 1
    if offset_ed > 0 and len(measures[i_ed]["event"]):
        for i_token, token in enumerate(measures[i_ed]["event"]):
            if token[0] == "o":
                if token2v(token) >= offset_ed:
                    flag = 0
                    break

        seg_measures[i_ed] = deepcopy(measures[i_ed])
        seg_measures[i_ed]["event"] = seg_measures[i_ed]["event"][:i_token + flag]

    # Remove redundant notes from the first measure
    if offset_st > 0 and len(measures[i_st]["event"]):
        for i_token, token in enumerate(measures[i_st]["event"]):
            if token[0] == "o":
                if token2v(token) >= offset_st:
                    break

        if i_token < len(measures[i_st]["event"]) - 1:
            seg_measures[i_st]["event"] = seg_measures[i_st]["event"][i_token:]
        else:
            seg_measures[i_st]["event"] = []
    return seg_measures
