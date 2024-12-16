import math
import numpy as np
from fractions import Fraction

from utils.common import normalize_tp


def get_t_bar(time_signature, tempo, **kwargs):
    """Get bar duration in seconds

    Args:
        time_signature (str): time signature, i.e. "4/4".
        tempo (str): tempo
    """
    # normalized_tempo = int(tempo / 12) * 12  # To avoid weird ticks
    normalized_tempo = normalize_tp(tempo)
    if len(time_signature.split('/')) > 2:
        time_signature = Fraction(
            time_signature[:-2]) / int(time_signature[-1])
    t_bar = Fraction(time_signature) * 4 * 60 / normalized_tempo
    return t_bar


def t_to_bar_beat(t, ts_shift, scale=6):
    """Convert time to bar/beat in unrolled midi

    Args:
        t (float): time
        ts_shift (list): A List of dictionary where each element marks the change in tempo and time signature, i.e.
            [{"measure": 0, "t": 0, "tempo", "tempo": 120, "time_signature": 4/4}]
        scale (int, optional): round to the 1/scale quarter note. Defaults to 6.

    Returns:
        _type_: _description_
    """

    n_shifts = len(ts_shift) - 1
    if t >= ts_shift[-2]['t']:
        i = n_shifts
    else:
        for i in range(1, n_shifts):
            if t <= ts_shift[i]['t']:
                break

    i -= 1
    tp = ts_shift[i]['tempo']
    ts = ts_shift[i]['time_signature']

    numerator = int(Fraction(ts) * 4)
    t_bar = get_t_bar(ts, tp)

    i_measure = ts_shift[i]['measure']

    # Round to closest measure
    round_thresh = (numerator - 1 / scale) / numerator
    n_measure = math.ceil((t - ts_shift[i]['t']) / t_bar - round_thresh)
    t_pos = t - ts_shift[i]['t'] - t_bar * n_measure
    i_measure += n_measure

    # Round to the closest 1/scale quarter note
    bins = np.arange(0, numerator, 1 / scale) * 60 / tp
    pos = Fraction(np.argmin(np.abs(t_pos - bins)), scale)

    return int(i_measure), pos


def bar_beat_to_t(pos, sig_shift):
    """Convert bar/beat to time in the unrolled midi.

    Args:
        pos (tuple): (bar, beat)
        sig_shift (list): _description_
    """

    # Find closest tempo/time signature shift
    shift_measure = np.array([e['measure'] for e in sig_shift])
    i_shift = np.where(pos[0] >= shift_measure)[0][-1]

    n_measure = pos[0] - sig_shift[i_shift]['measure']
    n_measure += pos[1] / Fraction(sig_shift[i_shift]['time_signature']) / 4
    t_diff = n_measure * get_t_bar(**sig_shift[i_shift])

    return sig_shift[i_shift]['t'] + t_diff
