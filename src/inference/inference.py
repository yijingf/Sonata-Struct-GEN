"""
inference.py

Description:
    This script generates a full section of music using the nextGEN and accGEN models.
    Given an initial melody phrase and a predefined sequence of phrase types (e.g., key modulation,
    regular continuation, or section ending), it performs the following steps:

    1. Generates the melody continuation using the nextGEN model.
    2. Concatenates all melody phrases into a complete section.
    3. Divides the melody into segments and generates the corresponding accompaniment using accGEN.
    4. Concatenates the accompaniment phrases into a full accompaniment section.
    5. Aligns the melody and accompaniment to produce a full score of the generated section.

Outputs:
    - MIDI files of individual generated phrases:
        <OUTPUT_DIR>/<test_case_id>/melody_phrase_*.mid
        <OUTPUT_DIR>/<test_case_id>/acc_phrase_*.mid
    - JSON files of generated token sequences:
        <OUTPUT_DIR>/<test_case_id>/melody_phrase_*.json
        <OUTPUT_DIR>/<test_case_id>/acc_phrase_*.json
    - Concatenated section:
        - Melody: 
            MIDI: <OUTPUT_DIR>/<test_case_id>/melody.mid
            JSON: <OUTPUT_DIR>/<test_case_id>/melody.json
        - Accompaniment:
            MIDI: <OUTPUT_DIR>/<test_case_id>/acc.mid
        - Full score:
            MIDI: <OUTPUT_DIR>/<test_case_id>/full_score.mid
"""

import os
import gc
import json
import torch
import pretty_midi
from copy import deepcopy
from fractions import Fraction
from transformers import EncoderDecoderModel

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tokenizer import BertTokenizer
from utils.event import trim_event
from utils.common import load_note_event
from utils.decode import decode_token_to_pm

# Constants
from utils.constants import DATA_DIR, MODEL_DIR, OUTPUT_DIR
melody_dir = os.path.join(DATA_DIR, "event_part", "norm_melody")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_phrase(phrase, pad_bar=True, max_measure_len=40):
    i_st, i_ed = min(phrase), max(phrase)
    ts = f"ts-{phrase[i_st]['time_signature']}"
    tp = f"tp-{phrase[i_st]['tempo']}"
    tokens = [ts, tp]
    for i, measure in phrase.items():
        tokens += measure['event']
        if pad_bar:
            if i > i_st and i < i_ed:
                n_pad = max_measure_len - len(measure['event'])
                tokens += ['sep' for _ in range(n_pad)]
        tokens += ['bar']
    tokens[-1] = 'eos'
    return tokens


def get_test_input(event, start_pos=(0, 0), end_pos=(8, 0)):
    segment = trim_event(event, start_pos, end_pos)

    encoder_input_tokens = flatten_phrase(segment)

    n_max = max(segment)
    decoder_segment = {i: segment[n_max + i - 1] for i in range(2)}
    decoder_input_tokens = flatten_phrase(decoder_segment)[:-1]  # remove eos

    res = {"encoder_input": encoder_input_tokens, "decoder_input": decoder_input_tokens}
    return res


def post_process_output(output, tokenizer):

    pred_tokens = tokenizer.convert_ids_to_tokens(output[0][1:].tolist())

    pred = []
    for i in pred_tokens:
        if i == 'eos':
            break
        else:
            pred.append(i)

    return pred


def split_by_measure(pred):
    tokens = []
    measure = []
    for token in pred[2:]:
        measure += [token]
        if token == 'bar':
            tokens.append(measure)
            measure = []
    tokens.append(measure)
    return tokens


def generate_melody(model, tokenizer, test_input, max_len=512):
    case_id = test_input["case_id"]
    os.makedirs(os.path.join(OUTPUT_DIR, case_id), exist_ok=True)

    # Initialize the section with the primer phrase
    ts_tp_token = deepcopy(test_input['encoder_input'][:2])
    res = split_by_measure(test_input["encoder_input"])

    # Save primer phrase
    pm = decode_token_to_pm(test_input["encoder_input"])
    os.makedirs(os.path.join(OUTPUT_DIR, case_id), exist_ok=True)
    midi_file = os.path.join(OUTPUT_DIR, case_id, "melody_phrase_0.mid")
    pm.write(midi_file)

    type_tokens = test_input["type_tokens"]
    pred = None

    for j, type_token in enumerate(type_tokens):
        if j == 0:
            input_ids = tokenizer.convert_tokens_to_ids(test_input['encoder_input'])
            decoder_input_tokens = [type_token] + test_input['decoder_input']
        else:
            measures = split_by_measure(pred)
            decoder_input_tokens = deepcopy(ts_tp_token) + measures[-2] + measures[-1]
            if decoder_input_tokens[-1] == 'bar':
                decoder_input_tokens = decoder_input_tokens[:-1]
            input_ids = tokenizer.convert_tokens_to_ids(pred)
            decoder_input_tokens = [type_token] + decoder_input_tokens

        decoder_input_ids = tokenizer.convert_tokens_to_ids(decoder_input_tokens)
        inputs = {
            "input_ids": torch.tensor(input_ids).unsqueeze(0).to(device),
            "decoder_input_ids": torch.tensor(decoder_input_ids).unsqueeze(0).to(device),
            "max_length": max_len
        }

        output = model.generate(**inputs)
        pred = post_process_output(output, tokenizer)

        # concatenate the generated output
        res = res[:-2] + split_by_measure(pred)  # overwrite last two measures with the new output

        pm = decode_token_to_pm(pred)
        midi_file = os.path.join(OUTPUT_DIR, case_id, f"melody_phrase_{j + 1}.mid")
        pm.write(midi_file)

        # token_file = os.path.join(OUTPUT_DIR, case_id, f"melody_phrase_{j + 1}.json")
        # with open(token_file, "w") as f:
        # json.dump(pred, f)

    # Final MIDI
    res_tokens = ts_tp_token
    for measure in res:
        res_tokens += measure
    pm = decode_token_to_pm(res_tokens)
    pm.write(os.path.join(OUTPUT_DIR, case_id, "melody.mid"))

    token_file = os.path.join(OUTPUT_DIR, case_id, f"melody.json")
    with open(token_file, "w") as f:
        json.dump(res_tokens, f)


def generate_accompaniment(model, tokenizer, case_id, n_acc_measure=7, max_len=512):

    # Load generated melody tokens and split into measures
    melody_file = os.path.join(OUTPUT_DIR, case_id, "melody.json")
    with open(melody_file) as f:
        tokens = json.load(f)
    measures = split_by_measure(tokens)
    ts_tp_token = deepcopy(tokens[:2])

    acc_measures = []
    acc_cnt = 0
    for j in range(0, len(measures), n_acc_measure):
        print(f"Generating accompaniment phrase {acc_cnt} from measure {j} to {j + 8}")
        segment_tokens = deepcopy(ts_tp_token)
        for k in range(j, min(j + 8, len(measures))):
            segment_tokens += measures[k]

        input_ids = tokenizer.convert_tokens_to_ids(segment_tokens[:max_len])
        decoder_input_ids = torch.tensor([tokenizer.mask_id]).unsqueeze(0)

        inputs = {
            "input_ids": torch.tensor(input_ids).unsqueeze(0).to(device),
            "decoder_input_ids": decoder_input_ids.to(device),
            "max_length": max_len
        }

        output = model.generate(**inputs)
        pred_tokens = tokenizer.convert_ids_to_tokens(output[0][1:].tolist())

        pred = []
        for token in pred_tokens:
            if token == 'eos':
                break
            pred.append(token)

        # Save each accompaniment phrase
        acc_file_base = os.path.join(OUTPUT_DIR, case_id, f"acc_{acc_cnt}")
        pm = decode_token_to_pm(pred)
        pm.write(f"{acc_file_base}.mid")

        acc_cnt += 1

        # Concatenate acc
        acc_measure = split_by_measure(tokens)
        acc_measures += acc_measures[:min(n_acc_measure, len(acc_measure))]

    final_tokens = deepcopy(ts_tp_token)
    for measure in acc_measures:
        final_tokens += measure

    pm = decode_token_to_pm(final_tokens)
    pm.write(os.path.join(OUTPUT_DIR, case_id, "acc.mid"))


def merge_track(melody_pm, acc_pm):

    # Create a new PrettyMIDI object for the merged result
    merged = pretty_midi.PrettyMIDI()

    # Add instruments from melody
    for inst in melody_pm.instruments:
        merged.instruments.append(inst)

    # Add instruments from accompaniment
    for inst in acc_pm.instruments:
        merged.instruments.append(inst)
    return


def main(test_cases, nextGEN_model_path, accGEN_model_path, max_len=512):

    # Load tokenizer
    base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
    tokenizer = BertTokenizer()
    tokenizer.load_base_vocab(base_vocab_file)

    # Generate melody first
    # Load model
    nextGEN_model = EncoderDecoderModel.from_pretrained(nextGEN_model_path).to(device).eval()

    for test_case in test_cases:
        event_file = os.path.join(melody_dir, test_case["event_file"])
        event = load_note_event(event_file)
        test_input = get_test_input(event,
                                    (0, 0),
                                    (test_case["end"]["measure"], test_case["end"]["pos"]))
        test_input["case_id"] = test_case["case_id"]
        test_input["type_tokens"] = test_case["type_tokens"]

        generate_melody(nextGEN_model, tokenizer, test_input, max_len)

    # Generate accompaniment
    # Avoid OOM
    del nextGEN_model
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    accGEN_model = EncoderDecoderModel.from_pretrained(accGEN_model_path).to(device).eval()

    for test_case in test_cases:
        generate_accompaniment(accGEN_model, tokenizer, test_case["case_id"], max_len)
        pm = pretty_midi.PrettyMIDI()

    # Merge melody and acc
    for test_case in test_cases:
        # Load the two MIDI files
        melody_file = os.path.join(OUTPUT_DIR, test_case["case_id"], "melody.mid")
        melody_pm = pretty_midi.PrettyMIDI(melody_file)

        acc_file = os.path.join(OUTPUT_DIR, test_case["case_id"], "acc.mid")
        acc_pm = pretty_midi.PrettyMIDI(acc_file)
        pm = merge_track(melody_pm, acc_pm)

        # Write the combined MIDI file
        pm.write(os.path.join(OUTPUT_DIR, test_case["case_id"], "full_score.mid"))


if __name__ == "__main__":

    nextGEN_model_path = os.path.join(MODEL_DIR, "nextGEN")
    accGEN_model_path = os.path.join(MODEL_DIR, "accGEN")

    # Define multiple test inputs here
    test_cases = [
        {
            "case_id": "01",
            "event_file": "beethoven/sonata03-4.json",
            "start": {"measure": 0, "pos": Fraction(1, 2)},
            "end": {"measure": 8, "pos": "1"},
            "type_tokens": ["key_mod", 'next', 'next', 'next', 'sect-end']
        },
        # Add more test cases here
    ]
    main(test_cases, nextGEN_model_path, accGEN_model_path)
