import os
import json
from glob import glob
from copy import deepcopy

import sys
sys.path.append("..")
from utils.event import expand_score
from utils.common import load_event
from utils.decode import decode_token_to_event, encode_event_to_token


def extract_acc(event, melody_event, dummy_ts_tp=["ts-4/4", "tp-120"]):

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


if __name__ == "__main__":

    data_dir = "../../sonata-dataset-phrase"

    event_dir = os.path.join(data_dir, "event")
    melody_dir = os.path.join(data_dir, "event_skyline")
    acc_dir = os.path.join(data_dir, "acc_no_repeat")

    composers = ['mozart', 'haydn', 'beethoven', 'scarlatti']
    for composer in composers:

        os.makedirs(os.path.join(acc_dir, composer), exist_ok=True)

        for event_file in sorted(glob(os.path.join(event_dir, composer, "*.json"))):

            base_name = os.path.basename(event_file)

            # Load full-score
            score, struct = load_event(event_file)
            event, _ = expand_score(score, struct, "no_repeat")

            # Load melody
            melody_file = os.path.join(melody_dir, composer, base_name)
            melody_score, struct = load_event(melody_file)
            melody_event, _ = expand_score(melody_score, struct, "no_repeat")

            acc_event = extract_acc(event, melody_event)
            with open(os.path.join(acc_dir, composer, base_name), "w") as f:
                json.dump(acc_event, f)
