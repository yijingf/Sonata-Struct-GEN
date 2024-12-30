"""
Quantized melody/skyline: quantize note onset/duration to 16th note
"""
import os
import json
from glob import glob
from copy import deepcopy
from fractions import Fraction

import sys
sys.path.append("..")
from utils.common import load_event
from utils.event import Event, expand_score


def quantize_token(val, step_per_beat=4):
    """Round onset/duration to closest subdivision of quarter note, defaults to 16th note

    Args:
        val (fractions.Fraction): onset/duration value
        step_per_beat (int, optional): n subdivisions of a quarter note. Defaults to 4.

    Returns:
        quantized_val (fractions.Fraction): quantized onset/duration value
    """
    frac = Fraction(round((val - int(val)) * step_per_beat), step_per_beat)
    quantized_val = frac + int(val)
    if val != 0 and quantized_val == 0:
        return Fraction(1, step_per_beat)  # Avoid quantizing duration to 0
    else:
        return quantized_val


def split_sects(event, cpts):
    sect_event = {}

    for i, entry in enumerate(cpts[:-1]):
        sect_event[str(i)] = {}
        measure_onset = entry['measure']
        for i_measure in range(measure_onset, cpts[i + 1]['measure']):
            sect_event[str(i)][i_measure - measure_onset] = deepcopy(event[i_measure])

    return sect_event


def quantize(event, step_per_beat=4):
    for i in range(len(event)):

        quantized_tokens = []
        last_onset = -1

        for token in event[i]['event']:
            e = Event(token)
            if e.event_type in ['onset', 'duration']:
                quantized_val = quantize_token(e.val, step_per_beat)

                # Remove notes with duplicated quantized onset
                if e.event_type == 'onset':
                    if quantized_val <= last_onset:
                        continue
                    else:
                        last_onset = quantized_val
                quantized_tokens.append(f"{token[0]}-{quantized_val}")
            else:
                quantized_tokens.append(token)

        event[i]['event'] = quantized_tokens

    return event


def main(fname):

    base_name = os.path.basename(fname)

    # Quantize Melody
    score, mark = load_event(fname)
    melody, _ = expand_score(score, mark, "no_repeat")
    quantized_melody = quantize(melody, step_per_beat=4)

    # Save quantized unfolded melody
    with open(os.path.join(quantized_event_dir, composer, base_name), "w") as f:
        json.dump(quantized_melody, f)

    return


if __name__ == "__main__":

    data_dir = "../../sonata-dataset-phrase"
    melody_dir = os.path.join(data_dir, "event_skyline")

    quantized_event_dir = os.path.join(data_dir, "quantized_skyline_no_repeat")
    os.makedirs(quantized_event_dir, exist_ok=True)

    for composer in ['beethoven', 'mozart', 'haydn', 'scarlatti']:

        os.makedirs(quantized_event_dir, "composer")

        for melody_file in glob(os.path.join(melody_dir, composer, f"*.json")):
            main(melody_file)
