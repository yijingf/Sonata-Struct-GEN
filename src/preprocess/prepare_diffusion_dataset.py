"""_summary_

Beat
"""

import os
import numpy as np
from glob import glob
from fractions import Fraction

import sys
sys.path.append("..")

from preprocess.diffusion_quantize import quantize
from preprocess.extract_melody import skyline_variation
from preprocess.extract_accompaniment import extract_acc
from utils.event import expand_score, event_to_pm, Event
from utils.common import load_event, normalize_event
from utils.decode import decode_token_to_event


# Estimate beat-level chord using CASSET from DECIBAL
sys.path.append("../../../DECIBEL")
from decibel.midi_chord_recognizer.event import Event as DecibelEvent
from decibel.music_objects.chord_annotation import ChordAnnotation
from decibel.music_objects.chord_vocabulary import ChordVocabulary

chord_vocabulary = ChordVocabulary.generate_chroma_major_minor()

CHORD_DEGREE = {
    "N": [],
    "C:maj": [0, 4, 7],
    "C:min": [0, 3, 7],
    "C#:maj": [1, 5, 8],
    "C#:min": [1, 4, 8],
    "D:maj": [2, 6, 9],
    "D:min": [2, 5, 9],
    "Eb:maj": [3, 7, 10],
    "Eb:min": [3, 6, 10],
    "E:maj": [4, 8, 11],
    "E:min": [4, 7, 11],
    "F:maj": [0, 5, 9],
    "F:min": [0, 5, 8],
    "F#:maj": [1, 6, 10],
    "F#:min": [1, 6, 9],
    "G:maj": [2, 7, 11],
    "G:min": [2, 7, 10],
    "Ab:maj": [3, 8, 0],
    "Ab:min": [3, 8, 11],
    "A:maj": [4, 9, 1],
    "A:min": [4, 9, 0],
    "Bb:maj": [5, 10, 2],
    "Bb:min": [5, 10, 1],
    "B:maj": [6, 11, 3],
    "B:min": [6, 11, 2]
}

CHORD_MAP = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}

CHORD_ROOT = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "Ab": 8,
    "A": 9,
    "Bb": 10,
    "B": 11
}


def parse_chord(chord_str):
    if chord_str == "N":
        return chord_str

    if ":" in chord_str:
        chord, mode = chord_str.split(":")
    else:
        chord = chord_str
        mode = "maj"

    norm_chord = CHORD_MAP.get(chord, chord)

    return ":".join([norm_chord, mode])


def get_midi_chord_annotation(scored_annotation_items):
    midi_chord_annotation = ChordAnnotation()

    current_annotation, _ = scored_annotation_items[0]
    last_added = False
    for annotation_item, _ in scored_annotation_items:
        if annotation_item.chord == current_annotation.chord:
            current_annotation.to_time = annotation_item.to_time
            last_added = False
        else:
            midi_chord_annotation.add_chord_annotation_item(current_annotation)
            current_annotation = annotation_item
            last_added = True
    if not last_added:
        midi_chord_annotation.add_chord_annotation_item(current_annotation)

    return midi_chord_annotation


def get_beat_events(pm, tempo, n_digit=4):
    t_beat = 60 / tempo
    beats = np.round(np.arange(0, pm.get_end_time() + t_beat, t_beat),
                     n_digit).tolist()

    # Create events
    events = dict()
    for i in range(0, len(beats) - 1):
        events[beats[i]] = DecibelEvent(beats[i], beats[i + 1])

    # Add each note to the corresponding events
    for instrument in pm.instruments:
        if not instrument.is_drum:
            # We assert that instrument.notes are ordered on start value for each instrument
            start_index = 0
            for note in instrument.notes:
                # Find suitable start_index
                while start_index < len(beats) - 1 and note.start >= events[beats[start_index]].end_time:
                    start_index += 1
                # Add this note to each event during which it sounds
                last_index = start_index
                events[beats[last_index]].add_note(note)
                while last_index < len(beats) - 1 and note.end > events[beats[last_index]].end_time:
                    last_index += 1
                    # Todo: A temporary fix
                    if last_index not in beats:
                        break
                    events[beats[last_index]].add_note(note)

    # Normalize each event
    for event_key in events:
        events[event_key].normalize()

    return events


def export_chord_annotation(chord_annotation, tempo, fname):

    # replace first blank chord with the first valid chord
    i = 0
    while i < len(chord_annotation.chord_annotation_items):
        chord_item = chord_annotation.chord_annotation_items[i]
        if chord_item.chord:
            break
        i += 1
    chord_annotation.chord_annotation_items[i].from_time = 0

    t_beat = 60 / tempo

    finalized_chords = []
    for chord_item in chord_annotation.chord_annotation_items[i:]:
        chord_str = str(chord_item.chord)
        if chord_str == 'None':
            chord_str = 'N'
        chord = parse_chord(chord_str)
        duration = int((chord_item.to_time - chord_item.from_time) / t_beat)

        if chord == "N":
            finalized_chords[-1][-1] += duration
            continue

        root_degree = CHORD_ROOT[chord.split(':')[0]]
        finalized_chords += [[chord, CHORD_DEGREE[chord], root_degree, duration]]

    with open(fname, "w") as f:
        for entry in finalized_chords:
            f.write(f"{entry[0]} {entry[1]} {entry[2]} {entry[3]}\n")

    return


def estimate_chord(pm, tempo, output_file):
    events = get_beat_events(pm, tempo)
    most_likely_chords = [events[event_key].find_most_likely_chord(chord_vocabulary)
                          for event_key in events.keys()]

    # Compute average chord probability
    concatenated_annotation = get_midi_chord_annotation(most_likely_chords)
    export_chord_annotation(concatenated_annotation, tempo, output_file)
    return


def export_notes(event, ts, output_file, step_per_beat=4):
    """Quantize note to 16th note, and convert to a overall onset duration representation.

    Args:
        event (_type_): _description_
        ts (_type_): _description_
        output_file (str): _description_
        step_per_beat (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    dummy_ts_tp = ["ts-4/4", "tp-120"]
    prev_step = 0
    notes = []

    for i in range(max(event)):
        bar_notes = decode_token_to_event(dummy_ts_tp + event[i]['event'])[-1][0]
        for note in bar_notes:
            onset = int((note[0] + i * ts * 4) * step_per_beat)
            pitch = Event(note[1]).val
            duration = int(note[2] * step_per_beat)

            if duration < 1:
                if not len(notes) or onset != notes[-1][0]:
                    duration = 1
                else:
                    continue

            if onset < prev_step:
                notes[-1][-1] = onset - notes[-1][0]
            elif onset > prev_step:
                notes.append([prev_step, 0, onset - prev_step])

            notes.append([onset, pitch, duration])
            prev_step = onset + duration

    with open(output_file, "w") as f:
        for note in notes:
            f.write(f"{note[1]} {note[2]}\n")
    return


def main(event_file, output_dir, estimate_chord_from_acc=False):

    composer = os.path.basename(os.path.dirname(event_file))
    basename = os.path.basename(event_file).split('.')[0]
    title = f"{composer}-{basename}"
    os.makedirs(os.path.join(output_dir, title), exist_ok=True)

    score_event, mark = load_event(event_file)
    orig_event, _ = expand_score(score_event, mark)

    # Key transposition
    event = normalize_event(orig_event, adjust_ts_tp=False)
    melody = skyline_variation(event)

    if estimate_chord_from_acc:
        acc = extract_acc(event, melody)
        pm, cpt = event_to_pm(acc)
    else:
        pm, cpt = event_to_pm(event)

    if len(cpt) > 1:
        print(f"Skip {title}: multiple tempo/time signature.")
        return

    os.makedirs(os.path.join(output_dir, title), exist_ok=True)

    # Chord Estimation
    chord_file = os.path.join(output_dir, title, "finalized_chord.txt")
    estimate_chord(pm, cpt[0]['tempo'], chord_file)

    # Melody Representation
    ts = Fraction(cpt[0]['time_signature'])
    quantized_melody = quantize(melody, step_per_beat=4)
    melody_file = os.path.join(output_dir, title, "melody.txt")
    export_notes(quantized_melody, ts, melody_file, step_per_beat=4)

    # Save Tempo
    tempo_file = os.path.join(output_dir, title, "tempo.txt")
    with open(tempo_file, "w") as f:
        f.write(str(cpt[0]['tempo']))
    return


if __name__ == "__main__":
    from glob import glob
    from tqdm import tqdm

    output_dir = "../../whole-song-gen-data/"
    for composer in ['haydn', 'beethoven', 'mozart', 'scarlatti']:
        event_files = sorted(
            glob(f"../../sonata-dataset-phrase/event/{composer}/*.json"))
        for event_file in tqdm(event_files, desc=composer):
            main(event_file, output_dir)
