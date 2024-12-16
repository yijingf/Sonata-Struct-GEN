import os
from glob import glob
from fractions import Fraction

import sys
sys.path.append("..")

from utils.common import load_event
from utils.event import expand_score, event_to_pm, Event


def convert_representation(event, ts, step_per_beat=4):
    """Quantize note to 16th note, and convert to a overall onset duration representation.

    Args:
        event (_type_): _description_
        ts (_type_): _description_
        step_per_beat (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """

    prev_step = 0
    notes = []

    for i_bar in range(max(event)):
        bar_event = event[i_bar]['event']
        for i in range(0, len(bar_event), 3):
            onset = int((Event(bar_event[i]).val + i_bar * ts * 4) * step_per_beat)
            pitch = Event(bar_event[i + 1]).val
            duration = int(Event(bar_event[i + 2]).val * step_per_beat)

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
    return notes


composers = ['mozart', 'beethoven', 'scarlatti', 'haydn']
event_dir = "../../sonata-dataset-phrase/event_skyline"
output_root_dir = "../../../whole-song-gen/sonata_data/"

for composer in composers:
    event_files = sorted(glob(os.path.join(event_dir, f"{composer}/*.json")))
    for event_file in event_files:
        score_event, struct = load_event(event_file)
        event, _ = expand_score(score_event, struct, repeat_mode="no_repeat")

        # Render to midi with quantized tempo
        pm, cpt = event_to_pm(event, quantize_tp=True)

        # Skip movements with tempo/time signature change
        if len(cpt) > 1:
            print(composer, os.path.basename(event_file))
            continue

        ts = Fraction(cpt[0]['time_signature'])
        notes = convert_representation(event, ts)

        basename = os.path.basename(event_file).split('.')[0]
        output_dir = os.path.join(output_root_dir, f"{composer}-{basename}")
        os.makedirs(output_dir, exist_ok=True)

        # Save Melody
        output_fname = os.path.join(output_dir, "melody.txt")
        with open(output_fname, "w") as f:
            for note in notes:
                f.write(f"{note[1]} {note[2]}\n")

        # Save Tempo
        output_fname = os.path.join(output_dir, "tempo.txt")
        with open(output_fname, "w") as f:
            f.write(str(cpt[0]['tempo']))

        # Save Time Signature
        # output_fname = os.path.join(output_dir, "time_signature.txt")
        # with open(output_fname, "w") as f:
        #     f.write(cpt[0]['time_signature'])
