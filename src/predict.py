import os
import json
import torch
import numpy as np
from copy import deepcopy

from midi_utils.rel_tokenizer import RelTokenizer
from utils.tokenizer import decode_token_to_pm, BertTokenizer
from utils.common import normalize_tp, pitch_transpose
from utils.event import flatten_measures, map_measure_to_token

# CONSTANT
from utils.constants import PITCH_OFFSET_DICT


def reset_velocity(pm):
    for note in pm.instruments[0].notes:
        note.velocity = 75
    return


def normalize_orig_event(event, note_key='original_note'):
    # Normalize event
    ts = event['original_time_signature']
    tp = normalize_tp(event['original_tempo'])
    event['time_signature'] = f"ts-{ts}"
    event['tempo'] = f"tp-{tp}"

    # Pitch Transpose
    pitch_offset = PITCH_OFFSET_DICT[event['key'].split()[0]]
    measures = deepcopy(event[note_key])

    for i, measure in enumerate(measures):
        for j, token in enumerate(measure):
            if token[0] not in ['o', 'd']:
                measures[i][j] = pitch_transpose(token, pitch_offset)

    event['note'] = measures
    return event


class EventPreprocessor():

    def __init__(self, tokenizer, max_len=512, device=None):
        self.tokenizer = tokenizer

        self.max_len = max_len
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def get_masked_token_ids(
            self, orig_event, mask_measure=[2, 3, 4, 5, 6],
            pad_bar=True):
        event = normalize_orig_event(orig_event)
        tokens = flatten_measures(deepcopy(event),
                                  eos_token=self.tokenizer.eos_token,
                                  pad_bar=pad_bar,
                                  bar_eos_token=self.tokenizer.sep_token)

        ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))

        idxs = np.append(2, np.where(np.array(tokens) == 'bar')[0] + 1)
        idxs = np.append(idxs, len(tokens))
        mask_idx = map_measure_to_token(idxs, mask_measure)

        ids[mask_idx] = self.tokenizer.mask_id
        return ids

    def get_primer_token_ids(self, primer_event, pad_bar=False):
        event = normalize_orig_event(primer_event, note_key='primer_note')
        tokens = flatten_measures(deepcopy(event),
                                  eos_token=self.tokenizer.eos_token,
                                  pad_bar=pad_bar,
                                  bar_eos_token=self.tokenizer.sep_token,
                                  add_eos=False)
        ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))
        return ids

    def prepare_inputs(self, token_ids):
        # key_padding_mask = torch.ones((1, max_len))
        # key_padding_mask[input_ids == pad_id] = 0
        # key_padding_mask[input_ids == mask_id] = 0
        input_ids = torch.tensor(token_ids).unsqueeze(0)
        inputs = {"input_ids": input_ids.to(self.device)}

        return inputs

    def prepare_encoder_decoder_inputs(self, token_ids):
        # key_padding_mask = torch.ones((1, max_len))
        # key_padding_mask[input_ids == pad_id] = 0
        # key_padding_mask[input_ids == mask_id] = 0
        inputs = torch.tensor(token_ids).unsqueeze(0)
        decoder_input_ids = inputs.clone()
        inputs = {"inputs": inputs.to(self.device),
                  "decoder_input_ids": decoder_input_ids.to(self.device),
                  "max_length": self.max_len}

        return inputs

    def prepare_mass_inputs(self, token_ids, mask_pad=False):
        len_pad = self.max_len - len(token_ids)

        if len_pad < 0:
            return None

        input_ids = torch.tensor(token_ids.copy(), dtype=torch.long)
        input_ids = torch.nn.functional.pad(
            input_ids, [0, len_pad],
            value=self.tokenizer.mask_id).unsqueeze(0)

        len_primer = np.where(token_ids == self.tokenizer.mask_id)[0][0]
        decoder_input_ids = torch.tensor(token_ids[:len_primer],
                                         dtype=torch.long).unsqueeze(0)

        inputs = {"inputs": input_ids.to(self.device),
                  "decoder_input_ids": decoder_input_ids.to(self.device),
                  "max_length": self.max_len}

        return inputs


def load_model(model_path, model_type):
    if model_type == 'mass' or model_type == 'encoder-decoder':
        from transformers import EncoderDecoderModel
        model = EncoderDecoderModel.from_pretrained(model_path)
    elif model_type == 'music-transformer':
        from models import MusicTransformer
        model = MusicTransformer.from_pretrained(model_path)
    elif model_type == 'bert':
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(model_path)

    return model


def main(
        model_path, model_type, event_files, mask_measure=[2, 3, 4, 5, 6],
        max_len=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, model_type)
    model.to(device)
    model.eval()

    # Load Tokenizer
    base_vocab_file = "../sonata-dataset/vocab/base_vocab.txt"
    with open(base_vocab_file) as f:
        base_vocab = f.read().splitlines()
    tokenizer = BertTokenizer()
    tokenizer.train(base_vocab)
    preprocessor = EventPreprocessor(tokenizer, max_len)

    for event_file in event_files:
        # output_file = os.path.join()
        with open(event_file) as f:
            orig_event = json.load(f)

        ids = preprocessor.get_masked_tokens_ids(orig_event,
                                                 mask_measure=mask_measure,
                                                 pad_bar=True)
        inputs = preprocessor.prepare_mass_inputs(ids)

        if model_type == 'mass':
            output = model.generate(**inputs)


# def predict(tokenizer, primer_file):

#     with open(primer_file) as f:
#         primer = json.load(f)
#     primer = normalize_primer(primer)

#     model_type = "bert"

#     if model_type == 'bert':
#         from transformers import BertForMaskedLM
#         from utils.tokenizer import BertTokenizer

#         model_file = "../../models/bert-phrase-512"
#         model = BertForMaskedLM.from_pretrained(model_file)

#         token_logits = model(input_ids).logits

#         # Find the location of [MASK] and extract its logits
#         mask_token_index = torch.where(input_ids == tokenizer.mask_id)[1]
#         mask_token_logits = token_logits[0, mask_token_index, :]

#         output_ids = input_ids.clone()
#         output_ids[0, mask_token_index] = torch.argmax(
#             mask_token_logits, dim=1)

#     elif model_type == 'music-transformer':

#         from models import MusicTransformer
#         from utils.tokenizer import BaseTokenizer

#         model_file = "../models/music-transformer-phrase-256/"
#         model = MusicTransformer.from_pretrained(model_file)

#         tokenizer = BaseTokenizer("../sonata-dataset/vocab/vocab_0804.txt")

#         max_len = 512
#         output_ids = input_ids.clone()
#         for _ in range(max_len - primer_len):
#             logits = model(output_ids)['logits']
#             pred = torch.argmax(logits, axis=-1)
#             output_ids = torch.concat([output_ids, pred[:, -1:]], axis=1)

#     output_ids = output_ids.detach().cpu().tolist()
#     pm = tokenizer.decode(output_ids)

#     output_fname = "x.mid"
#     pm.write(output_fname)
