import os
import json
import pandas as pd

from utils.common import normalize_tp, normalize_ts_tp, normalize_pitch
from utils.tokenizer import BertTokenizer


def validate_phrase(phrase, tokenizer, seq_len=512):
    normed_ts = phrase['time_signature']
    normed_tp = normalize_tp(phrase['tempo'])
    # normed_ts, normed_tp = normalize_ts_tp(phrase['time_signature'],
    #                                        phrase['tempo'])

    n_bar = len(phrase['event']) - 1
    len_phrase = sum([len(i) for i in phrase['event']])

    if len_phrase >= seq_len - 3 - n_bar:  # ts, tp, eos, bar line
        return None

    for measure in phrase['event']:
        # Remove invalid phrase with irregular onset position/duration
        if any([i not in tokenizer.vocab for i in measure if i[0] in ['o', 'd']]):
            return None

    # i_st = min(phrase)

    # token_ids = {"ts-tp":tokenizer.convert_tokens_to_ids([f"ts-{normed_ts}", f"tp-{normed_tp}"]),
    # "token_ids": [tokenizer.convert_tokens_to_ids(i) for i in phrase['event']]}
    # return token_ids

    tokens = {"key": phrase['key'].split()[0],
              "ts-tp": [f"ts-{normed_ts}", f"tp-{normed_tp}"],
              "tokens": [i for i in phrase['event']]}
    return tokens


# base_vocab_file = "../sonata-dataset-phrase/vocab/base_vocab.txt"
base_vocab_file = "../sonata-dataset-phrase/vocab/base_vocab_no_pitch.txt"
with open(base_vocab_file) as f:
    base_vocab_no_pitch = f.read().splitlines()
tokenizer_no_pitch = BertTokenizer(add_sep=False)
tokenizer_no_pitch.train(base_vocab_no_pitch)
