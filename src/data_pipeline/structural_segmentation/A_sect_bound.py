# air/variation and # measures < 20, use section boundary markers from score as boundary
# from struct_segment import *

import numpy as np

import sys
sys.path.append("..")
from utils.align import get_t_bar, t_to_bar_pos, bar_pos_to_t


def naive_t_bound(pred_bound, k=2):
    t_boundaries, labels = pred_bound[k]
    t_boundaries = t_boundaries[:, 0]
    labels = np.array(labels)
    t_recur = t_boundaries[np.array(labels == labels[0])]

    if len(t_recur) < 2:
        return None, None

    elif len(t_recur) == 2:
        # sonata
        t_recap = t_recur[1]
        t_dev = t_boundaries[np.where(t_boundaries == t_recap)[0][0] - 1]
        if t_dev == 0:
            t_dev = None
    else:
        # rondo
        t_recap = t_recur[2]
        t_last_entry = t_recur[1]
        t_dev = t_boundaries[np.where(t_boundaries == t_last_entry)[0][0] - 1]
    return t_dev, t_recap


def get_sect_bound_by_sig_cpt(sig_cpt):
    # Signature change points potentially mark section boundaries
    entry = sig_cpt[0]
    ts_tp_key = (entry['tempo'], entry['time_signature'], entry['key'])

    flag = False
    for i, entry in enumerate(sig_cpt[2: -1]):
        if ts_tp_key == (entry['tempo'], entry['time_signature'], entry['key']):
            flag = True
            break

    if flag:
        t_sect = {"expose": []}
        for j, expose_entry in enumerate(sig_cpt[: i + 1]):
            t_sect['expose'] += [[expose_entry['t'], sig_cpt[j + 1]['t']]]
        t_dev = sig_cpt[i + 1]['t']
        t_recap = entry['t']
        t_sect['dev'] = [[t_dev, t_recap]]
        t_sect['recap'] = []
        for j, entry in enumerate(sig_cpt[i + 2: -1]):
            t_sect["recap"] += [[entry['t'], sig_cpt[j + i + 3]['t']]]

        return t_sect

        # t_expose = sig_cpt[0]['t']
        # t_dev = sig_cpt[i]['t']  # the before entry before the entry with same ts_tp
        # t_recap = entry['t']
        # t_fin = sig_cpt[-1]['t']
        # return {"expose": [[t_expose, t_dev]],
        #         "dev": [[t_dev, t_recap]],
        #         "recap": [[t_recap, t_fin]]}

    return None


def get_t_third_entry(t_recur):
    """
    rondo
    second and third entry of first subject
    """
    diff = np.diff(t_recur[:, 0])
    diff_ratio = diff / max(diff)

    i_third_occr = 0

    cnt = 0
    for i, v in enumerate(np.array(diff_ratio > 0.3)):
        if v:
            cnt += 1
        if cnt == 2:
            i_third_occr = i
            break

    return t_recur[i_third_occr + 1][0]


def analyze_t_recur(t_recur):
    """Get t recapiulation and t last entry

    Args:
        t_recur (_type_): _description_

    Returns:
        _type_: _description_
    """

    # t_last_entry: the second entry of first subject in the first section
    if len(t_recur) < 2:
        t_recap = 0
        t_last_entry = None
    elif len(t_recur) == 2:
        # probably sonata with only one recurrence of first subject
        t_last_entry = None
        t_recap = t_recur[-1, 0]
    elif len(t_recur) <= 4:
        # probably rondo with only multiple recurrence of first subject
        t_last_entry = t_recur[1, 0]
        t_recap = t_recur[2, 0]
    else:
        t_recap = get_t_third_entry(t_recur)
        t_last_entry = (t_recur[np.where(t_recur[:, 0] == t_recap)[0] - 1])[0][0]
    return t_last_entry, t_recap


def estimate_t_dev(t_last_entry, t_boundaries, labels):
    """Estimate development section boundary based on the last entry of the first subject before recapitulation.

    Args:
        t_last_entry (_type_): _description_
        t_boundaries (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx = np.where(t_boundaries[:, 0] == t_last_entry)[0][0]
    idx_first = np.where(labels == labels[idx])[0][0]
    i = 0
    while labels[idx + i] == labels[idx_first + i]:
        i += 1
    t_dev = t_boundaries[idx + i][0]  # or [1]?
    return t_dev


def get_struct_A(pred_bound, sig_cpt):

    t_dev = None
    t_expose = sig_cpt[0]['t']
    t_fin = sig_cpt[-1]['t']
    t_boundaries, labels = pred_bound[-1]
    labels = np.array(labels)

    if len(sig_cpt) > 2:
        t_sect = get_sect_bound_by_sig_cpt(sig_cpt)
        if t_sect is not None:
            return t_sect
        else:
            # An exception: signature shifts occur in coda.
            t_fin = sig_cpt[1]['t']
            idx = t_boundaries[:, 0] < t_fin
            t_boundaries = t_boundaries[idx]
            labels = labels[idx]

    t_recur = t_boundaries[labels == labels[0]]
    t_last_entry, t_recap = analyze_t_recur(t_recur)

    if t_recap == 0 or t_recap / t_fin > 0.9:
        t_bar = get_t_bar(**sig_cpt[0])
        if np.diff(t_boundaries[0]) < 2 * t_bar:
            t_recur = t_boundaries[labels == labels[1]]
            t_last_entry, t_recap = analyze_t_recur(t_recur)
            if t_recap == 0:
                t_dev, t_recap = naive_t_bound(pred_bound)
        else:
            t_dev, t_recap = naive_t_bound(pred_bound)

    if t_last_entry and t_dev is None:
        t_dev = estimate_t_dev(t_last_entry, t_boundaries, labels)
        if t_dev is None:
            recap_idx = np.where(t_boundaries[:, 0] == t_recap)[0][0]
            for i, label in enumerate(labels[1: recap_idx]):
                n_repeat = np.sum(labels == label)
                if n_repeat == 1:
                    t_dev = t_boundaries[i + 1, 0]
                    break

    if t_recap is None:
        return {"part i": [[t_expose, t_fin]]}

    # Align with beat
    t_recap = bar_pos_to_t(t_to_bar_pos(t_recap, sig_cpt), sig_cpt)

    if t_dev is None:
        return {"expose": [[t_expose, t_recap]],
                "recap": [[t_recap, t_fin]]}
    else:
        t_dev = bar_pos_to_t(t_to_bar_pos(t_dev, sig_cpt), sig_cpt)
        t_sect = {"expose": [[t_expose, t_dev]],
                  "dev": [[t_dev, t_recap]],
                  "recap": [[t_recap, t_fin]]}
        if t_fin == sig_cpt[-1]['t']:
            return t_sect

        for entry in sig_cpt[2:]:
            t_sect['recap'] += [[t_sect['recap'][-1][-1], entry['t']]]

        return t_sect
