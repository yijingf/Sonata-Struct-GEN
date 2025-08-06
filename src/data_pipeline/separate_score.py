"""
separate_score.py

Description:
    Separates the melody (skyline) from the accompaniment in symbolic music files. 
    Also stores the rendered MIDI from the full score, along with time signature and tempo change points.

    Input files are located in:
        - DATA_DIR/event/<composer>
    Output is saved in:
        - DATA_DIR/event_part/melody/<composer>
        - DATA_DIR/event_part/acc/<composer>
        - DATA_DIR/midi/<composer>
Usage:
    python3 separate_score.py
"""

import os
import json
import tqdm
from copy import deepcopy
from fractions import Fraction

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import load_event, get_file_list
from utils.event import Event, expand_score, event_to_pm
from utils.decode import decode_token_to_event, encode_event_to_token

# Constants
from utils.constants import DATA_DIR, COMPOSERS
dummy_ts_tp = ["ts-4/4", "tp-120"]


def skyline(event, duration_first=True):
    """Only keep note with shortest duration and highest pitch"""

    melody_event = {}

    for i_measure in sorted(event.keys()):

        # Extract melody for each measure
        melody_event[i_measure] = deepcopy(event[i_measure])
        measure = []
        pairs = []

        for token in event[i_measure]['event']:
            e = Event(token)

            if e.event_type == 'onset':

                if len(pairs):
                    if duration_first:
                        note = sorted(pairs,
                                      key=lambda x: (x[1].val, -x[0].val))[0]
                    else:
                        note = sorted(pairs, key=lambda x: -x[0].val)[0]
                    measure += [note[0].token_str, note[1].token_str]

                pairs = []
                measure += [e.token_str]

            elif e.event_type == 'pitch':
                pairs += [[e]]

            elif e.event_type == 'duration':
                pairs[-1] += [e]

        if len(pairs):
            if duration_first:
                note = sorted(pairs, key=lambda x: (x[1].val, -x[0].val))[0]
            else:
                note = sorted(pairs, key=lambda x: -x[0].val)
            melody_event[i_measure]['event'] = measure + [note[0].token_str,
                                                          note[1].token_str]

    return melody_event


def skyline_variation(event):
    """ Only keep note with shortest duration and highest pitch; ignoring the lower-pitch notes that start after its onset and before its offset."""

    melody_event = {}

    offset = 0
    measure_onset = -Fraction(event[min(event)]['time_signature']) * 4
    last_pitch = None
    note_onset = 0

    # for i_measure in range(0, 14):
    for i_measure in sorted(event.keys()):

        len_bar = Fraction(event[i_measure]['time_signature']) * 4

        # Extract melody for each measure
        melody_event[i_measure] = deepcopy(event[i_measure])
        measure = []
        pairs = []  # pitch-duration pair

        for token in event[i_measure]['event']:
            e = Event(token)

            if e.event_type == 'onset':

                if len(pairs):
                    # find the highest pitch at one onset
                    note = sorted(pairs, key=lambda x: -x[0].val)[0]

                    note_onset = measure_onset + onset.val

                    long_note = offset - note_onset >= len_bar  # no new notes
                    if last_pitch:
                        neighbor_pitch = abs(note[0].val - last_pitch) <= 12
                        dist = long_note and neighbor_pitch
                    else:
                        dist = False

                    # ignore low-pitch note that start before the offset of the last note
                    if not last_pitch or note[0].val > last_pitch or note_onset >= offset or dist:

                        measure += [onset.token_str,
                                    note[0].token_str,
                                    note[1].token_str]

                        if note_onset + note[1].val >= offset:
                            last_pitch = note[0].val

                        if offset - note_onset >= len_bar:
                            offset = note_onset + note[1].val
                        else:
                            offset = max(note_onset + note[1].val, offset)

                onset = e
                pairs = []

            elif e.event_type == 'pitch':
                pairs += [[e]]

            elif e.event_type == 'duration':
                pairs[-1] += [e]

        if len(pairs):
            note = sorted(pairs, key=lambda x: -x[0].val)[0]

            long_note = offset - note_onset >= len_bar  # no new notes
            if last_pitch:
                neighbor_pitch = abs(note[0].val - last_pitch) <= 12
                dist = long_note and neighbor_pitch
            else:
                dist = False

            if not last_pitch or note[0].val > last_pitch or measure_onset + onset.val >= offset or dist:
                measure += [onset.token_str,
                            note[0].token_str,
                            note[1].token_str]

                if measure_onset + onset.val + note[1].val >= offset:
                    last_pitch = note[0].val

                if offset - note_onset >= len_bar:
                    offset = note_onset + note[1].val
                else:
                    offset = max(note_onset + note[1].val, offset)

        melody_event[i_measure]['event'] = measure
        measure_onset += len_bar

    return melody_event


def melody_extract_pitch(event):
    """ Only keep highest pitch of the right hand part"""

    melody_event = {}

    for i_measure in sorted(event.keys()):

        # Extract melody for each measure
        melody_event[i_measure] = deepcopy(event[i_measure])
        melody_measure = []
        pitch_duration = []

        for token in event[i_measure]['event']:
            e = Event(token)

            if e.event_type == 'onset':

                if len(pitch_duration):
                    pitch_duration_str = [pitch_duration[0].token_str,
                                          pitch_duration[1].token_str]
                else:
                    pitch_duration_str = []

                melody_measure += pitch_duration_str
                pitch_duration = []

                melody_measure += [e.token_str]
                add_note = True

            elif e.event_type == 'pitch':

                if not len(pitch_duration):
                    pitch_duration = [e]

                elif e.val > pitch_duration[0].val:
                    pitch_duration = [e]
                    add_note = True
                else:
                    add_note = False

            elif e.event_type == 'duration':

                if add_note:
                    pitch_duration += [e]

        if len(pitch_duration):
            pitch_duration_str = [pitch_duration[0].token_str,
                                  pitch_duration[1].token_str]
        else:
            pitch_duration_str = []
        melody_event[i_measure]['event'] = melody_measure + pitch_duration_str

    return melody_event


def extract_acc(event, melody_event):

    acc_event = deepcopy(event)

    for i in range(len(event)):
        notes = decode_token_to_event(dummy_ts_tp + event[i]['event'])[-1][0]
        melody_notes = decode_token_to_event(
            dummy_ts_tp + melody_event[i]['event'])[-1][0]

        acc_notes = []
        for note in notes:
            if note not in melody_notes:
                acc_notes.append(note)

        acc_event[i]['event'] = encode_event_to_token([acc_notes])

    return acc_event


def main():
    event_dir = os.path.join(DATA_DIR, "event")
    melody_dir = os.path.join(DATA_DIR, "event_part", "melody")
    acc_dir = os.path.join(DATA_DIR, "event_part", "acc")
    midi_dir = os.path.join(DATA_DIR, "midi")

    for composer in COMPOSERS:
        os.makedirs(os.path.join(melody_dir, composer), exist_ok=True)
        os.makedirs(os.path.join(acc_dir, composer), exist_ok=True)
        os.makedirs(os.path.join(midi_dir, composer), exist_ok=True)

    for event_file in tqdm(sorted(get_file_list(event_dir))):
        try:
            score, mark = load_event(event_file)
            event, idx_mapping = expand_score(score, mark, repeat_mode="no_repeat")

            # render to midi with normalized tempo
            mapping_file = event_file.replace(event_dir, midi_dir)
            midi_file = mapping_file.replace(".json", ".mid")

            pm, cpt = event_to_pm(event, quantize_tp=True)
            pm.write(midi_file)

            with open(mapping_file, "w") as f:
                json.dump({"idx_mapping": idx_mapping, "cpt": cpt}, f)

            melody_event = skyline_variation(event)
            acc_event = extract_acc(event, melody_event)

            melody_file = event_file.replace(event_dir, melody_dir)
            with open(melody_file, "w") as f:
                json.dump(melody_event, f)

            acc_file = event_file.replace(event_dir, acc_dir)
            with open(acc_file, "w") as f:
                json.dump(acc_event, f)

        except:
            print(f"Failed. {event_file}")

    return


if __name__ == "__main__":
    main()
