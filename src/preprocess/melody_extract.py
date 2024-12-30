import os
import json
from copy import deepcopy
from fractions import Fraction

import sys
sys.path.append("..")
from utils.event import Event


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


if __name__ == "__main__":

    from glob import glob
    from tqdm import tqdm
    from utils.common import load_event
    from utils.event import expand_score, event_to_pm

    repeat_mode = "no_repeat"
    DATA_DIR = "../../sonata-dataset-phrase"

    orig_event_dir = os.path.join(DATA_DIR, "event")
    event_dir = os.path.join(DATA_DIR, "event_skyline")
    midi_dir = os.path.join(DATA_DIR, "skyline_midi_no_repeat")

    for composer in ['mozart', 'haydn', 'beethoven', 'scarlatti']:

        os.makedirs(os.path.join(event_dir, composer), exist_ok=True)
        os.makedirs(os.path.join(midi_dir, composer), exist_ok=True)

        event_files = sorted(
            glob(os.path.join(orig_event_dir, composer, "*.json")))

        for orig_event_file in tqdm(event_files, desc=f"Process {composer}"):
            basename = os.path.basename(orig_event_file).split(".")[0]

            event_file = os.path.join(event_dir, composer, f"{basename}.json")
            midi_file = os.path.join(midi_dir, composer, f"{basename}.mid")
            mapping_file = os.path.join(midi_dir, composer, f"{basename}.json")

            try:
                event, mark = load_event(orig_event_file)
                melody_score = skyline_variation(event)
                with open(event_file, "w") as f:
                    json.dump({"note": melody_score, "mark": mark}, f)

                # Render event to midi
                melody_event, idx_mapping = expand_score(melody_score,
                                                         mark,
                                                         repeat_mode="no_repeat")

                # render to midi with quantized tempo
                pm, cpt = event_to_pm(melody_event, quantize_tp=True)
                pm.write(midi_file)
                with open(mapping_file, "w") as f:
                    json.dump({"idx_mapping": idx_mapping, "cpt": cpt}, f)

            except:
                print(f"Failed. {event_file}")
