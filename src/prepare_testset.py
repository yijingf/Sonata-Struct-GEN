"""Prepare test data. 

Output:

Usage: 
"""
import os
from copy import deepcopy
from fractions import Fraction

from utils.tokenizer import BertTokenizer
from utils.decode import decode_token_to_pm
from utils.common import trim_event, load_event, normalize_tp

from midi_utils.common import change_pitch
from midi_utils.rel_tokenizer import RelTokenizer

# Constant
from utils.constants import PITCH_OFFSET_DICT, DATA_DIR


def check_consistency(phrase):
    # sanity check
    tempo, time_signature, key = set(), set(), set()

    for i in phrase:
        tempo.add(phrase[i]['tempo'])
        time_signature.add(phrase[i]['time_signature'])
        key.add(phrase[i]['key'])

    if any([len(key) > 1, len(tempo) > 1, len(time_signature) > 1]):
        raise ValueError("Inconsistent key, tempo or time signature")

    return


def prepare_event(event, start=(0, 0), end=(8, 0), tempo_to_bin=True):

    # Get note
    seg_event = trim_event(event, start, end)
    notes = [seg_event[i]['event']
             for i in range(start[0], max(seg_event) + 1)]

    # Decode note events to midi
    # norm_ts = normalize_ts(event[start[0]]['time_signature'])
    # ts_token = f"ts-{norm_ts}"
    ts = event[start[0]]['time_signature']
    ts_token = f"ts-{ts}"

    tp = deepcopy(event[start[0]]['tempo'])
    if tempo_to_bin:
        tp = normalize_tp(tp)
    tp_token = f"tp-{tp}"

    tokens = [ts_token, tp_token]
    for measure_token in notes:
        tokens += measure_token
        tokens += ['bar']
    pm = decode_token_to_pm(tokens)
    return notes, pm


def main(info, base_vocab_file):
    output_dir = os.path.join(DATA_DIR, "primer_event")

    # Load Krn Tokenizer
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)

    # Load Magenta's Relative MIDI-like Tokenizer
    rel_tokenizer = RelTokenizer(num_velocity_bins=1, add_eos=False)

    event_dir = os.path.join(DATA_DIR, "event")

    for item_id, item in info.items():

        event_file = os.path.join(event_dir, item['event_file'])
        start = (item["start"]['measure'], Fraction(item['start']['pos']))
        end = (item["end"]['measure'], Fraction(item['end']['pos']))

        event, _ = load_event(event_file)

        ts = event[start[0]]['time_signature']
        tp = event[start[0]]['tempo']
        key = event[start[0]]['key']

        # Entire sequence
        notes, pm = prepare_event(event, start, end)
        for measure_notes in notes:
            if tokenizer.has_irregular_token(measure_notes):
                raise ValueError("Irregular tokens")
        pm.write(os.path.join(output_dir, f"{item_id}.mid"))

        # 2-bar primer
        primer_end = (start[0] + 2, start[1])
        primer_notes, primer_pm = prepare_event(event, start, primer_end)

        # Pitch transpose to C Major/Minor
        pitch_offset = PITCH_OFFSET_DICT[key.split()[0]]
        change_pitch(primer_pm, pitch_shift=int(pitch_offset), inplace=True)
        primer_pm.write(os.path.join(output_dir, f"primer_{item_id}_in_C.mid"))

        # Encode primer using magenta's relative MIDI-like tokenizer
        primer_rel_tokens = rel_tokenizer.encode_pm(primer_pm)

        res = {"original_time_signature": ts,
               "original_tempo": tp,
               "key": key,
               "original_note": notes,
               "primer_note": primer_notes,
               "primer_rel_tokens": primer_rel_tokens}

        with open(os.path.join(output_dir, f"{item_id}.json"), "w") as f:
            json.dump(res, f)


if __name__ == '__main__':
    import json

    with open("../sonata-dataset/primer_event/info.json") as f:
        event_info = json.load(f)

    # event_info = {
    #     "01": {"event_file": "mozart/sonata09-3-1.json",
    #            "start": {"measure": 0, "pos": '3/2'},
    #            "end": {"measure": 8, "pos": '3/2'}},
    #     "02": {"event_file": "mozart/sonata09-3-1.json",
    #            "start": {"measure": 48, "pos": '0'},
    #            "end": {"measure": 55, "pos": '3'}},
    #     "03": {"event_file": "haydn/sonata61-1.json",
    #            "start": {"measure": 11, "pos": '0'},
    #            "end": {"measure": 18, "pos": '4'}},
    #     "04": {"event_file": "haydn/sonata34-1.json",
    #            "start": {"measure": 12, "pos": '3/2'},
    #            "end": {"measure": 20, "pos": '3/2'}},
    #     "05": {"event_file": "beethoven/sonata11-1.json",
    #            "start": {"measure": 4, "pos": '0'},
    #            "end": {"measure": 11, "pos": '3'}},
    #     "06": {"event_file": "beethoven/sonata09-1.json",
    #            "start": {"measure": 5, "pos": '0'},
    #            "end": {"measure": 12, "pos": '4'}},
    #     "07": {"event_file": "scarlatti/L127K348.json",
    #            "start": {"measure": 36, "pos": '0'},
    #            "end": {"measure": 43, "pos": '3'}},
    #     "08": {"event_file": "scarlatti/L166K085.json",
    #            "start": {"measure": 16, "pos": '0'},
    #            "end": {"measure": 23, "pos": '4'}},
    #     "09": {"event_file": "mozart/sonata02-1.json",
    #            "start": {"measure": 1, "pos": '0'},
    #            "end": {"measure": 8, "pos": '3'}},
    #     "10": {"event_file": "mozart/sonata15-3.json",
    #            "start": {"measure": 8, "pos": '1'},
    #            "end": {"measure": 16, "pos": '1'}},
    #     "11": {"event_file": "mozart/sonata07-1.json",
    #            "start": {"measure": 15, "pos": '0'},
    #            "end": {"measure": 22, "pos": '4'}},
    #     "12": {"event_file": "mozart/sonata15-2.json",
    #            "start": {"measure": 40, "pos": '1'},
    #            "end": {"measure": 48, "pos": '1'}},
    #     "13": {"event_file": "beethoven/sonata26-1.json",
    #            "start": {"measure": 21, "pos": '0'},
    #            "end": {"measure": 28, "pos": '4'}},
    #     "14": {"event_file": "beethoven/sonata26-3.json",
    #            "start": {"measure": 33, "pos": '0'},
    #            "end": {"measure": 40, "pos": '3'}},
    #     "15": {"event_file": "beethoven/sonata18-3.json",
    #            "start": {"measure": 0, "pos": '2'},
    #            "end": {"measure": 8, "pos": '2'}},
    #     "16": {"event_file": "beethoven/sonata24-2-0.json",
    #            "start": {"measure": 57, "pos": '1/2'},
    #            "end": {"measure": 65, "pos": '1/2'}},
    #     "17": {"event_file": "beethoven/sonata07-4.json",
    #            "start": {"measure": 73, "pos": '0'},
    #            "end": {"measure": 80, "pos": '4'}},
    #     "18": {"event_file": "scarlatti/L306K345.json",
    #            "start": {"measure": 14, "pos": '2'},
    #            "end": {"measure": 22, "pos": '2'}},
    #     "19": {"event_file": "scarlatti/L348K244.json",
    #            "start": {"measure": 15, "pos": '0'},
    #            "end": {"measure": 22, "pos": '3/2'}},
    #     "20": {"event_file": "scarlatti/L350K498.json",
    #            "start": {"measure": 40, "pos": '0'},
    #            "end": {"measure": 47, "pos": '3'}},
    #     "21": {"event_file": "beethoven/sonata24-2-0.json",
    #            "start": {"measure": 74, "pos": '3/2'},
    #            "end": {"measure": 82, "pos": '3/2'}},
    # }

    main(event_info)
