import warnings
import pretty_midi
from fractions import Fraction

from utils.common import token2v

# Constants
DEFAULT_TIME_SIGNATURE = "4/4"
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 75


def pitch_name_to_pm_pitch(pitch_name):

    if "-" in pitch_name:
        pitch_name = "".join(pitch_name.split("-"))
        pitch = pretty_midi.note_name_to_number(pitch_name) - 1
    else:
        pitch = pretty_midi.note_name_to_number(pitch_name)

    return pitch


def decode_time_signature(ts_str):
    ts_digit = ts_str.split("/")
    if len(ts_digit) > 2:
        ts_num = Fraction(int(ts_digit[0]), int(ts_digit[1]))
    else:
        ts_num = int(ts_digit[0])
    ts_denom = int(ts_digit[-1])
    return ts_num, ts_denom


def decode_ts_tp(tokens):

    ts, tp = None, None

    for token in tokens:

        if token[:2] == "ts":
            ts = token[3:]
            # ts = decode_time_signature(token[3:])
        elif token[:2] == "tp":
            tp = int(token[3:])

        if ts and tp:
            break

    return ts, tp


def encode_event_to_token(events):

    tokens = []
    for i, measure in enumerate(events):
        last_onset = -1
        for onset, pitch, duration in measure:
            if onset == last_onset:
                tokens += [pitch, f"d-{duration}"]
            else:
                tokens += [f'o-{onset}', pitch, f"d-{duration}"]
                last_onset = onset
        if i < len(events) - 1:
            tokens += ['bar']
    return tokens


def decode_token_to_event(tokens, bar_eos_token="sep"):
    ts, tp = decode_ts_tp(tokens)

    if not ts or not tp:
        raise ValueError("Time signature/tempo not found.")
    # ts_num, _ = ts
    bar_dur = Fraction(ts) * 4

    events = {}
    notes = []

    onset = None
    bar = 0
    events[bar] = []

    seq_len = len(tokens)

    i = 0
    decode_flag = True

    while i < seq_len:
        token = tokens[i]
        i += 1

        if token == "bar":
            decode_flag = True
            onset = Fraction(0)

            for note in notes:
                dur = bar_dur - note[0]
                events[bar].append(note + [dur])

            bar += 1
            events[bar] = []
            notes = []
            continue

        if not decode_flag:
            continue

        if bar_eos_token and token == bar_eos_token:
            decode_flag = False
            continue

        elif token[:2] in ["ts", "tp"]:
            continue

        elif token[0] == "o":
            onset = token2v(token)

        elif token[0] == "d":
            dur = token2v(token)
            for note in notes:
                events[bar].append(note + [dur])

            notes = []

        else:
            notes.append([onset, token])

    return ts, tp, events


def decode_token_to_pm(tokens, bar_eos_token="sep", bar_limit=False, t_offset=0):
    """_summary_

    Args:
        tokens (list): _description_
        bar_eos_token (str, optional): _description_. Defaults to "sep".
        bar_limit (bool, optional): _description_. Defaults to False.
        t_offset (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    ts, tp, events = decode_token_to_event(tokens, bar_eos_token)

    if ts is None:
        warnings.warn(f"No time signature. Set to {DEFAULT_TIME_SIGNATURE}")
        ts = DEFAULT_TIME_SIGNATURE
        # ts_frac = Fraction(DEFAULT_TIME_SIGNATURE)
        # ts = (ts_frac.numerator, ts_frac.denominator)

    if not tp:
        warnings.warn(f"No tempo. Set to {DEFAULT_TEMPO}")
        tp = DEFAULT_TEMPO

    t_quarter_note = 60 / tp

    bar_dur = Fraction(ts) * 4
    # ts_num, ts_denom = ts
    # assert ts_denom == 4

    inst = pretty_midi.Instrument(program=0)
    n_bar = len(events)
    for i in range(n_bar):
        for onset, pitch_name, duration in events[i]:

            # if bar_limit and onset >= ts_num:
            if bar_limit and onset >= bar_dur:
                continue

            # t_st = (onset + i * ts_num) * t_quarter_note + t_offset
            t_st = (onset + i * bar_dur) * t_quarter_note + t_offset
            t_ed = t_st + duration * t_quarter_note
            pitch = pitch_name_to_pm_pitch(pitch_name)
            note = pretty_midi.Note(DEFAULT_VELOCITY, pitch, t_st, t_ed)
            inst.notes.append(note)

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    return pm
