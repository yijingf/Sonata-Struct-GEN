"""
Segment events in DATA_DIR/event/composer into phrases with a fixed length and hop-size in DATA_DIR/segment/composer.

We assume that time signature and tempo would not change within a phrase. Therefore, we start a new phrase once a time signature and tempo change is identified. 

All musical events are normalized:
* Notes are transposed to C major or minor. 
* Time signature are normalized to a denominator of 4.
* Tempo are digitized into 10 bins

Usage: 
# Segment into 8-bar phrases with 2-bar hop size
python3 segment.py [--len_phrase 8] [--hop_size 2]

"""

import json
from fractions import Fraction

import sys
sys.path.append("..")
from utils.event import concat_event, remove_repeat, get_sub_sect_event
from utils.common import load_event, normalize_ts_tp, normalize_event, token2v

# Constants
from utils.constants import DATA_DIR


def merge_section_name(pattern):
    """Merge section names with same prefix, such as A, A1, A2 into one section

    Args:
        pattern (_type_): _description_

    Returns:
        _type_: _description_
    """
    merged_sect_name = []
    merged_sect = []

    last_sect = pattern[0]
    for sect in pattern:
        if sect[0] == last_sect:
            merged_sect.append(sect)
        else:
            merged_sect_name.append(merged_sect)
            merged_sect = [sect]
        last_sect = sect[0]

    if merged_sect:
        merged_sect_name.append(merged_sect)

    return merged_sect_name


def phrase_segment(measures, max_len_phrase=8, measure_offset=0, phrase_offset=0):
    """Segment sections into phrases containing certain number of measures.

    Args:
        measures (_type_): _description_
        max_len_phrase (int, optional): Max number of measures per phrase. Defaults to 8.
        measure_offset (int, optional): Segment offset. Defaults to 0.
        phrase_offset (int, optional): Phrase count offset. Defaults to 0.

    Returns:
        _type_: _description_
    """
    min_i_measure = min(measures) + measure_offset
    max_i_measure = max(measures)

    if measure_offset >= max_i_measure:
        return None

    measure = measures[min_i_measure]
    ts = measure['time_signature']
    tp = measure['tempo']

    normed_ts, normed_tp = normalize_ts_tp(ts, tp)
    ts_token = f"ts-{normed_ts}"
    tp_token = f"tp-{normed_tp}"

    phrases = {}

    i_phrase = phrase_offset
    offset = token2v(measure['event'][0])
    len_phrase = 1 - offset / Fraction(ts) / 4

    phrases[i_phrase] = {"time_signature": ts_token,
                         "tempo": tp_token,
                         # "key": [measure["key"]],
                         "note": [measure['event']]}

    for i in range(min_i_measure + 1, max_i_measure + 1):

        notes = measures[i]['event']

        # Start a new phrase if time signature changes
        if measures[i]['time_signature'] != ts:

            # update time signature, tempo, offset
            ts = measures[i]['time_signature']
            tp = measures[i]['tempo']

            normed_ts, normed_tp = normalize_ts_tp(ts, tp)
            ts_token = f"ts-{normed_ts}"
            tp_token = f"tp-{normed_tp}"

            offset = token2v(notes[0])

            # start a new phrase
            len_phrase = 1 - token2v(notes[0]) / Fraction(ts) / 4
            i_phrase += 1
            phrases[i_phrase] = {"time_signature": ts_token,
                                 "tempo": tp_token,
                                 "note": [notes],
                                 "key": [measures[i]["key"]]}

            continue

        # Phrase Boundary
        if len_phrase + 1 <= max_len_phrase:
            if i_phrase not in phrases:
                phrases[i_phrase] = {"time_signature": ts_token,
                                     "tempo": tp_token,
                                     "key": [],
                                     "note": []}

            phrases[i_phrase]['note'] += [notes]
            # phrases[i_phrase]['key'] += [measures[i]['key']]
            len_phrase += 1

        else:
            curr_notes = []
            # Add tokens to current phrase
            i_token = 0
            for i_token, token in enumerate(notes):

                if token[0] != 'o':
                    curr_notes.append(token)
                else:
                    onset = token2v(token)
                    if onset < offset:
                        curr_notes.append(f"o-{onset}")
                    else:
                        break

            if curr_notes:
                phrases[i_phrase]['note'] += [curr_notes]
            # phrases[i_phrase]['key'] += [measures[i]['key']]

            # Start a new phrase with rest of the tokens in the current measure
            i_phrase += 1

            if i_token < len(notes) - 1:
                offset = token2v(notes[i_token])
                len_phrase = 1 - token2v(notes[0]) / Fraction(ts) / 4

                phrases[i_phrase] = {"time_signature": ts_token,
                                     "tempo": tp_token,
                                     #  "key": [measures[i]['key']],
                                     "note": [notes[i_token:]]}

            # No token left in current measure
            else:
                offset = 0
                len_phrase = 0

    return phrases


def get_sect_event(event_file):
    # Load note event, and structure
    event, struct = load_event(event_file)

    # Remove repetition if the section has no first/second volta
    norep_pattern = remove_repeat(struct['pattern'])

    # Sort sub sections by onset
    onsets = sorted([(i, v) for i, v in struct['attr'].items()],
                    key=lambda x: x[1]['idx'])
    sub_sect_event = get_sub_sect_event(event, onsets)

    # Merge events within sections such as [A, A1, A, A2]
    merged_pattern = merge_section_name(norep_pattern)

    # Get phrases for merged sections
    sect_event = {}
    for sects in merged_pattern:
        sect_event[sects[0]] = concat_event(sub_sect_event,
                                            sects,
                                            struct['attr'])[0]

    return sect_event


def segment(sect_event=None, event_file=None, max_len_phrase=8, hop_size=4):

    if sect_event is None:
        if event_file is None:
            raise ValueError(f"Invalid file name {event_file}.")
        sect_event = get_sect_event(event_file)

    # Get phrases from section
    all_phrases = []
    for event in sect_event.values():

        # Key, time transpose
        event = normalize_event(event)

        for measure_offset in range(0, max_len_phrase, hop_size):
            phrases = phrase_segment(event, max_len_phrase, measure_offset)
            if not phrases:
                continue
            for phrase in phrases.values():
                all_phrases.append(phrase)

    return all_phrases


if __name__ == "__main__":
    import os
    import json
    import argparse
    from glob import glob
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--len_phrase", dest="len_phrase", type=int,
                        default=8, help="Max phrase length.")
    parser.add_argument("--hop_size", dest="hop_size", type=int,
                        default=2, help="Hop size.")
    args = parser.parse_args()

    event_dir = os.path.join(DATA_DIR, "event")
    seg_dir = os.path.join(DATA_DIR, "segment")

    composers = os.listdir(event_dir)
    for composer in composers:  # 'mozart', 'haydn', 'beethoven', 'scarlatti'

        print(composer)
        os.makedirs(os.path.join(seg_dir, composer), exist_ok=True)

        event_files = glob(os.path.join(event_dir, composer, "*.json"))

        for event_file in tqdm(event_files):

            basename = os.path.basename(event_file).split(".")[0]
            seg_file = os.path.join(seg_dir, composer, f"{basename}.json")

            # Segment events into phrases
            segments = segment(event_file=event_file,
                               max_len_phrase=args.phrase_len,
                               hop_size=args.hop_size)

            with open(seg_file, "w") as f:
                json.dump(segments, f)
