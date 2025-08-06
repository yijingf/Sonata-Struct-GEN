import json
from tqdm import tqdm
from music21 import pitch
from utils.event import Event
from collections import Counter

from common import token2v, load_note_event


def norm_pitch_vocab(pitch_vocab, min_pitch=24, max_pitch=98, preset_pitch=False):
    if not preset_pitch:
        pitch_vals = [pitch.Pitch(i).ps for i in pitch_vocab]
        min_pitch = int(min(pitch_vals))
        max_pitch = int(max(pitch_vals))
    vocab = [pitch.Pitch(i).nameWithOctave for i in range(min_pitch, max_pitch + 1)]
    return sorted(vocab)


def clean_vocab(vocab_cnt, freq_thresh=50, min_pitch=24, max_pitch=98,
                preset_pitch=False):
    """Remove irregular note onset/duration

    Args:
        vocab_cnt (collections.Counter): _description_
    """
    orig_vocab = list(vocab_cnt.keys())
    for token in orig_vocab:
        e = Event(token)
        if e.event_type != "pitch":
            if e.val == 0:
                continue
            if not all([e.val.denominator % 5,
                        e.val.denominator % 7,
                        e.val.denominator % 11,
                        e.val.denominator % 12,  # new
                        e.val.denominator % 16,  # new
                        e.val.denominator % 24,
                        e.val.denominator % 32]):
                del vocab_cnt[e.token_str]

    pitch_vocab = [i for i in vocab_cnt if Event(i).event_type == "pitch"]
    vocab = set(norm_pitch_vocab(pitch_vocab, min_pitch, max_pitch, preset_pitch))

    vocab.update([i for i, v in vocab_cnt.items() if v >= freq_thresh])
    return list(vocab)


def build_vocab_from_event(event_file_list, filter=False, min_freq=20):

    vocab_cnt = Counter()
    ts_set = set()
    tp_set = set()

    for event_file in tqdm(event_file_list, desc="Build Vocabulary"):
        event = load_note_event(event_file)

        # Update vocabulary count
        for i in event:
            vocab_cnt.update(event[i]['event'])
            ts_set.add(f"ts-{event[i]['time_signature']}")
            tp_set.add(f"tp-{event[i]['tempo']}")

    # Post process vocabulary
    pitch_vocab = [i for i in vocab_cnt if Event(i).event_type == "pitch"]
    vocab = set(norm_pitch_vocab(pitch_vocab, preset_pitch=False))

    vocab.update([i for i in vocab_cnt])
    vocab.update(ts_set)
    vocab.update(tp_set)

    if filter:
        # Reject 1/11, 1/5, 1/7 notes or notes with occurences < 20
        tokens = list(vocab.keys())
        for token in tokens:
            if vocab[token] < min_freq:
                vocab.pop(token)
            elif token[0] in ['o', 'd']:
                div = token2v(token).denominator

                if not div % 5 or not div % 11 or not div % 7:
                    vocab.pop(token)

    base_vocab = sorted(list(vocab) + ["bar"])

    return base_vocab


def build_vocab_from_segment(seg_file_list, filter=False, min_freq=20):

    vocab_cnt = Counter()
    ts_set = set()
    tp_set = set()

    for seg_file in tqdm(seg_file_list, desc="Build Vocabulary"):

        with open(seg_file) as f:
            segments = json.load(f)

        # Update vocabulary count
        for seg in segments:
            for measure in seg['note']:
                vocab_cnt.update(measure)

            vocab_cnt.update([seg['time_signature']])
            vocab_cnt.update([str(seg['tempo'])])

    # Post process vocabulary
    pitch_vocab = [i for i in vocab_cnt if Event(i).event_type == "pitch"]
    vocab = set(norm_pitch_vocab(pitch_vocab, preset_pitch=False))

    vocab.update([i for i in vocab_cnt])
    vocab.update(ts_set)
    vocab.update(tp_set)

    if filter:
        # Reject 1/11, 1/5, 1/7 notes or notes with occurences < 20
        tokens = list(vocab.keys())
        for token in tokens:
            if vocab[token] < min_freq:
                vocab.pop(token)
            elif token[0] in ['o', 'd']:
                div = token2v(token).denominator

                if not div % 5 or not div % 11 or not div % 7:
                    vocab.pop(token)

    base_vocab = sorted(list(vocab) + ["bar"])

    return base_vocab
