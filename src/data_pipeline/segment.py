"""
segment.py

Description:
    Segments normalized event files (produced by normalize_notes.py) into fixed-length phrases
    for preparing a dataset for pretraining MASS. Also uses score-midi mapping files to detect time 
    signature and tempo change points that guide segmentation.

    Input by default:
        - Normalized melody event files:
            DATA_DIR/event_part/norm_melody/<composer>/
        - Score MIDI mapping files (used for tempo/time signature changes):
            DATA_DIR/midi/<composer>/
    Output:
        - Segmented phrases saved to:
            DATA_DIR/segment/<composer>/

Notes:
    It is assumed that time signature and tempo remain constant within a phrase.
    A new segment is started whenever a change in time signature or tempo is detected.

Usage:
    python3 segment.py [--event_dir DIR] [--cpt_dir DIR] [--seg_dir DIR] 
                       [--len_seg LEN] [--n_hop HOP]

Arguments:
    --event_dir   Directory of normalized melody event files. 
                  Defaults to DATA_DIR/event_part/norm_melody
    --cpt_dir     Directory of score MIDI mapping files.
                  Defaults to DATA_DIR/midi
    --seg_dir     Directory to save the segmented output.
                  Defaults to DATA_DIR/melody_segments
    --len_seg     Length of each segment in bars. (default: 8)
    --n_hop    Hop size between segments in bars. (default: 2)

Example:
    # Segment into 8-bar segments with 2-bar hop
    python3 segment.py --len_seg 8 --n_hop 2
"""
import os
import json
from tqdm import tqdm
from copy import deepcopy
from fractions import Fraction

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import trim_event, token2v, get_file_list, load_note_event

# Constants
from utils.constants import COMPOSERS, DATA_DIR


def get_sect_pickups(event, cpts):
    """Get pick ups for each sub-section

    Args:
        event (dict): Expanded score as is performed.
        cpts (list):  A list of time signature/tempo change points.

    Returns:
        pickups (list): A list of fraction
    """
    pickups = [token2v(event[entry['measure']]['event'][0]) for entry in cpts]
    return pickups


def split_sects(event, cpts):
    sect_event = {}

    for i, entry in enumerate(cpts[:-1]):
        sect_event[i] = {}
        measure_onset = entry['measure']
        for i_measure in range(measure_onset, cpts[i + 1]['measure']):
            sect_event[i][i_measure -
                          measure_onset] = deepcopy(event[i_measure])

    return sect_event


def segment(event, cpts, pickups=None, len_seg=8, n_hop=2):
    """Devide a piece into segments.

    Args:
        event (dict): The expanded score as performed (a dict of events).
        cpts (list): A list of time signature/tempo change points.
        pickups(list, optional): List of pickup measures or beats(if any). Defaults to None.
        len_seg (int, optional): Maximum segment length in bars. Defaults to 8.
        n_hop (int, optional): Hop size between two segments in bars. Defaults to 2.

    Returns:
        list: A list of segmented phrases(each segment being a list or dict of events).
    """

    # Sanity check
    assert cpts[-1]['measure'] == max(event) + 1

    if pickups is not None:
        assert len(pickups) == len(cpts) - 1
    else:
        pickups = get_sect_pickups(event, cpts[:-1])

    segments = []
    for i_sect, entry in enumerate(cpts[:-1]):

        pickup = pickups[i_sect]

        i_st = entry['measure']
        i_ed = cpts[i_sect + 1]['measure']
        n_beat = Fraction(event[i_st]['time_signature']) * 4
        ts_token = f"ts-{event[i_st]['time_signature']}"
        tp_token = f"tp-{event[i_st]['tempo']}"

        for i in range(i_st, i_ed, n_hop):

            start_pos = (i, pickup)

            if i + len_seg + pickup / n_beat <= i_ed:
                end_pos = (i + len_seg, pickup)
            else:
                end_pos = (i_ed, 0)

            segment = trim_event(event, start=start_pos, end=end_pos)
            keys = set([segment[j]['key'] for j in range(i, max(segment) + 1)])
            notes = [segment[j]['event'] for j in range(i, max(segment) + 1)]

            segments.append({'time_signature': ts_token,
                             'tempo': tp_token,
                             'key': list(keys),
                             "note": notes})

    return segments


def main(event_dir, cpt_dir, seg_dir, len_seg, n_hop):

    for composer in COMPOSERS:
        os.makedirs(os.path.join(seg_dir, composer), exist_ok=True)

    for event_file in tqdm(sorted(get_file_list(event_dir))):

        event = load_note_event(event_file)
        cpt_file = event_file.replace(event_dir, cpt_dir)
        with open(cpt_file) as f:
            cpts = json.load(f)['cpt']
        cpts.append({'measure': max(event) + 1})
        pickups = get_sect_pickups(event, cpts[:-1])

        segments = segment(event, cpts, pickups, len_seg, n_hop)

        seg_file = event_file.replace(event_dir, seg_dir)
        with open(seg_file, "w") as f:
            json.dump(segments, f)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    event_dir = os.path.join(DATA_DIR, "event")
    cpt_dir = os.path.join(DATA_DIR, "midi")
    seg_dir = os.path.join(DATA_DIR, "segments")

    parser.add_argument("--event_dir", type=str, default=event_dir,
                        help=f"Directory of event files to be segmented. Defaults to {event_dir}")
    parser.add_argument("--cpt_dir", type=str, default=cpt_dir,
                        help=f"Directory of score MIDI mapping files. Defaults to {cpt_dir}")
    parser.add_argument("--seg_dir", type=str, default=seg_dir,
                        help=f"Directory to output segmented files. Defaults to {seg_dir}")
    parser.add_argument("--len_seg", type=int, default=8,
                        help="Length of each segment in bars. Defaults to 8.")
    parser.add_argument("--n_hop", type=int, default=2,
                        help="Hop size between segments in bars. Defaults to 2.")
    args = parser.parse_args()

    main(args.event_dir, args.cpt_dir, args.seg_dir, args.len_seg, args.n_hop)
