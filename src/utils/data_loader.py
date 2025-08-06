import json
import torch
import random
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset


def _mask_collate_batch(examples, pad_id=0, mask_id=4, pred_masked_only=False):
    """
    Apply bar-level masking.
    """
    tokens, mask_idx = [], []
    for i, e in enumerate(examples):
        if len(e[1]):
            tokens.append(e[0])
            mask_idx.append(e[1])

    if isinstance(examples[0][0], (list, tuple, np.ndarray)):
        tokens = [torch.tensor(e, dtype=torch.long) for e in tokens]

    max_len = max(int(x.size(0)) for x in tokens)
    input_result = tokens[0].new_full([len(tokens), max_len], pad_id)
    label_result = tokens[0].new_full([len(tokens), max_len], pad_id)
    decoder_input = tokens[0].new_full([len(tokens), max_len], mask_id)

    for i in range(len(tokens)):
        seq_len = tokens[i].shape[0]
        input = tokens[i].clone()
        input[mask_idx[i]] = mask_id
        input_result[i, :seq_len] = input

        decoder_input[i, mask_idx[i]] = tokens[i][mask_idx[i]]

        if pred_masked_only:
            label_result[i, mask_idx[i]] = tokens[i][mask_idx[i]]
        else:
            label_result[i, :seq_len] = tokens[i]

    # see shift_tokens_right for decoder input
    pad = torch.tensor([[mask_id] for _ in range(len(tokens))], dtype=torch.long)
    decoder_input = torch.cat((pad, decoder_input[:, :-1]), 1)

    return input_result, decoder_input, label_result


class MassDataCollator():
    """
    Data collator
    """

    def __init__(self, mask_id=4, pad_id=0, pred_masked_only=False,
                 mask_pad=True, pad=True):

        self.mask_id = mask_id
        self.pad_id = pad_id
        self.pred_masked_only = pred_masked_only
        self.mask_pad = mask_pad
        self.pad = pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        if self.pad:
            inputs, decoder_inputs, labels = _mask_collate_batch(
                examples, pad_id=self.pad_id, mask_id=self.mask_id,
                pred_masked_only=self.pred_masked_only)
        else:
            inputs = _collate_batch(examples, pad_id=self.pad_id)
            decoder_inputs = inputs.clone()
            labels = inputs.clone()

        batch = {"input_ids": inputs}

        batch['decoder_input_ids'] = decoder_inputs

        if self.mask_pad:
            # Encoder Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

            # Decoder Attention Mask, equivalent to key_padding_mask
            # Causal Mask is used in huggingface EncoderDecoder Model by default
            decoder_attention_mask = torch.ones_like(decoder_inputs)
            decoder_attention_mask[decoder_inputs == self.pad_id] = 0
            batch['decoder_attention_mask'] = decoder_attention_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


def _dynamic_mask_collate_batch(examples, pad_id=0, mask_id=4,
                                vocab_size=324,
                                corrupt_ratio=.15, mask_ratio=.8, replace_ratio=.1):
    """
    Collate `examples` into a batch, adapted from huggingface transformer
    Randomly corrupt individual tokens in a sequence.
    """
    # Assume examples[0][1] is invalid Mask
    tokens = []
    for i, e in enumerate(examples):
        tokens.append(e[0])

    # Tensorize if necessary.
    if isinstance(tokens[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in tokens]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    max_len = max(x.size(0) for x in examples)

    input_ids = examples[0].new_full([len(examples), max_len], pad_id)
    labels = examples[0].new_full([len(examples), max_len], pad_id)

    for i, example in enumerate(examples):

        seq_len = example.shape[0]

        label = example.clone()

        n_corrupt = int(seq_len * corrupt_ratio)
        corrupt_idx = random.sample(range(seq_len), n_corrupt)

        n_mask = int(n_corrupt * mask_ratio)
        mask_idx = corrupt_idx[: n_mask]

        n_replace = int(n_corrupt * replace_ratio)
        replace_idx = corrupt_idx[n_mask: n_mask + n_replace]
        replaced = random.choices(range(pad_id + 1, vocab_size), k=n_replace)

        labels[i, replace_idx] = label[replace_idx]
        labels[i, mask_idx] = label[mask_idx]

        input_ids[i, :seq_len] = example
        input_ids[i, replace_idx] = torch.tensor(replaced, dtype=torch.long)
        input_ids[i, mask_idx] = mask_id

    return input_ids, labels


class BertDataCollator():
    """
    Bert Data Corruption
    The training data generator chooses 15% of the token positions at random for prediction. 
    If the i-th token is chosen, we replace the i-th token with 
    (1) the [MASK] token 80% of the time
    (2) a random token 10% of the time
    (3) the unchanged i-th token 10% of the time. 
    Then, Ti will be used to predict the original token with cross entropy loss. 
    """

    def __init__(self, pad_id=0, mask_id=4, max_len=512, vocab_size=324,
                 corrupt_ratio=.15, mask_ratio=.8, replace_ratio=.1, mask_pad=True):

        self.pad_id = pad_id
        self.mask_id = mask_id
        self.max_len = max_len

        self.vocab_size = vocab_size
        self.corrupt_ratio = corrupt_ratio

        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        self.mask_pad = mask_pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {}
        input_ids, labels = _dynamic_mask_collate_batch(
            examples, pad_id=self.pad_id, mask_id=self.mask_id,
            vocab_size=self.vocab_size, corrupt_ratio=self.corrupt_ratio,
            mask_ratio=self.mask_ratio, replace_ratio=self.replace_ratio)

        batch['input_ids'] = input_ids

        if self.mask_pad:
            # Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


def _collate_batch(examples, pad_id=0, max_len=512,
                   pad_to_max_len=False, align_right=False):
    """Collate `examples` into a batch, adapted from huggingface transformer"""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    if not pad_to_max_len:
        # Creating the full tensor and filling it with our data.
        max_len = max(x.size(0) for x in examples)

    result = examples[0].new_full([len(examples), max_len], pad_id)

    for i, example in enumerate(examples):

        if align_right:
            result[i, -example.shape[0]:] = example
        else:
            result[i, :example.shape[0]] = example

    return result


class BaseDataCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=0, q_len=None, max_len=256,
                 pad_to_max_len=False, align_right=False):
        self.q_len = q_len
        self.pad_id = pad_id
        self.max_len = max_len
        self.pad_to_max_len = pad_to_max_len
        self.align_right = align_right

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {"input_ids": _collate_batch(examples, pad_id=self.pad_id)}

        labels = batch["input_ids"].clone()

        if self.q_len is not None:
            labels = labels[:, -self.q_len:]
            batch["q_len"] = self.q_len

        key_padding_mask = torch.ones_like(labels)
        key_padding_mask[labels == self.pad_id] = 0
        batch['key_padding_mask'] = key_padding_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


class BaseDataset(Dataset):
    def __init__(self, token_path=None, input_tokens=None, seq_len=512, shuffle=True):

        with open(token_path) as f:
            phrases = json.load(f)

        if self.input_tokens != None:
            self.input_tokens = input_tokens
        elif token_path is None:
            raise ValueError("Invalid token_path or input_tokens.")
        else:
            self.input_tokens = [i['token_ids']
                                 for i in phrases if len(i['token_ids']) <= seq_len]

        if shuffle:
            idx = list(range(len(self.input_tokens)))
            random.shuffle(idx)
            self.input_tokens = [self.input_tokens[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        token_ids = self.input_tokens[index]
        return token_ids

    def __len__(self):
        return len(self.input_tokens)


class MaskedDataset(Dataset):
    def __init__(self, token_path, seq_len=512, shuffle=True, mask_mode='center'):

        with open(token_path) as f:
            phrases = json.load(f)

        phrases = [i for i in phrases if len(i['token_ids']) <= seq_len]

        self.input_tokens = [i['token_ids'] for i in phrases]
        if mask_mode == 'center':
            self.mask_idx = [i['center_mask_idx'] for i in phrases]
        elif mask_mode == 'rand':
            self.mask_idx = [i['rand_mask_idx'] for i in phrases]
        elif mask_mode == 'mix':
            self.input_tokens += self.input_tokens
            self.mask_idx = [i['rand_mask_idx']
                             for i in phrases] + [i['center_mask_idx'] for i in phrases]
        else:
            raise ValueError("Unknown masking type.")

        if shuffle:
            idx = list(range(len(self.input_tokens)))
            random.shuffle(idx)
            self.input_tokens = [self.input_tokens[i] for i in idx]
            self.mask_idx = [self.mask_idx[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        token_ids = self.input_tokens[index]
        mask_idx = self.mask_idx[index]
        return token_ids, mask_idx

    def __len__(self):
        return len(self.input_tokens)


# Next Phrase Generation
class BasePairDataset(Dataset):
    def __init__(self, token_path=None, max_len=512, shuffle=True):

        with open(token_path) as f:
            phrase_pairs = json.load(f)

        qualified_phrase_pairs = []
        for phrase_0, phrase_1 in phrase_pairs:
            if len(phrase_0) <= max_len and len(phrase_1) <= max_len:
                qualified_phrase_pairs.append((phrase_0, phrase_1))

        if shuffle:
            idx = list(range(len(qualified_phrase_pairs)))
            random.shuffle(idx)
            self.phrase_pairs = [qualified_phrase_pairs[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        phrase_0, phrase_1 = self.phrase_pairs[index]
        return phrase_0, phrase_1

    def __len__(self):
        return len(self.phrase_pairs)


def _base_pair_collate_batch(examples, pad_id=0):

    encoder_inputs, decoder_inputs = [], []
    for i, e in enumerate(examples):
        encoder_inputs.append(e[0])
        decoder_inputs.append(e[1])

    if isinstance(examples[0][0], (list, tuple, np.ndarray)):
        encoder_inputs = [torch.tensor(e, dtype=torch.long)
                          for e in encoder_inputs]
        decoder_inputs = [torch.tensor(e, dtype=torch.long)
                          for e in decoder_inputs]

    max_len_encoder = max(int(x.size(0)) for x in encoder_inputs)
    max_len_decoder = max(int(x.size(0)) for x in decoder_inputs)
    max_len = max(max_len_encoder, max_len_decoder)
    input = encoder_inputs[0].new_full([len(examples), max_len], pad_id)
    decoder_input = decoder_inputs[0].new_full(
        [len(examples), max_len], pad_id)

    for i in range(len(encoder_inputs)):
        encoder_seq_len = encoder_inputs[i].shape[0]
        input[i, :encoder_seq_len] = encoder_inputs[i]

        decoder_seq_len = decoder_inputs[i].shape[0]
        decoder_input[i, :decoder_seq_len] = decoder_inputs[i]

    return input, decoder_input


class BaseNextPhraseCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=0, pad=True, mask_pad=True):

        self.pad_id = pad_id
        self.pad = pad
        self.mask_pad = mask_pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        inputs, decoder_inputs = _base_pair_collate_batch(examples,
                                                          pad_id=self.pad_id)
        batch = {"input_ids": inputs}
        batch['decoder_input_ids'] = decoder_inputs

        if self.mask_pad:
            # Encoder Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

            # Decoder Attention Mask, equivalent to key_padding_mask
            # Causal Mask is used in huggingface EncoderDecoder Model by default
            decoder_attention_mask = torch.ones_like(decoder_inputs)
            decoder_attention_mask[decoder_inputs == self.pad_id] = 0
            batch['decoder_attention_mask'] = decoder_attention_mask

        # Shift decoder inputs one token left (remove the phrase_type token), and pad
        pad = torch.tensor([[self.pad_id] for _ in range(len(inputs))],
                           dtype=torch.long)
        labels = torch.cat((decoder_inputs.clone()[:, 1:], pad), 1)
        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


class MaskedPairDataset(Dataset):
    def __init__(self, token_path=None, shuffle=True):

        with open(token_path) as f:
            phrase_pairs = json.load(f)

        if shuffle:
            idx = list(range(len(phrase_pairs)))
            random.shuffle(idx)
            self.phrase_pairs = [phrase_pairs[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        phrase_1, phrase_2, phrase_3 = self.phrase_pairs[index]
        return phrase_1, phrase_2, phrase_3

    def __len__(self):
        return len(self.phrase_pairs)


def _acc_collate_batch(examples, pad_id=0, mask_id=4):
    """
    Apply bar-level masking.
    """
    melody, acc = [], []
    for i, e in enumerate(examples):
        if len(e[1]):
            melody.append(e[0])
            acc.append(e[1])

    if isinstance(examples[0][0], (list, tuple, np.ndarray)):
        melody = [torch.tensor(e, dtype=torch.long) for e in melody]
        acc = [torch.tensor(e, dtype=torch.long) for e in acc]

    melody_max_len = max(int(x.size(0)) for x in melody)
    acc_max_len = max(int(x.size(0)) for x in acc)

    n_sample = len(examples)
    input_result = melody[0].new_full([n_sample, melody_max_len], pad_id)
    label_result = acc[0].new_full([n_sample, acc_max_len], pad_id)
    decoder_input = acc[0].new_full([n_sample, acc_max_len], pad_id)

    for i in range(n_sample):
        input_result[i, :melody[i].shape[0]] = melody[i].clone()
        decoder_input[i, :acc[i].shape[0]] = acc[i].clone()
        label_result[i, :acc[i].shape[0]] = acc[i].clone()

    # right shift
    pad = torch.tensor([[mask_id] for _ in range(n_sample)], dtype=torch.long)
    decoder_input = torch.cat((pad, decoder_input[:, :-1]), 1)

    return input_result, decoder_input, label_result


class AccDataCollator():
    """
    Data collator
    """

    def __init__(self, mask_id=4, pad_id=0, mask_pad=True):

        self.mask_id = mask_id
        self.pad_id = pad_id
        self.mask_pad = mask_pad

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        inputs, decoder_inputs, labels = _acc_collate_batch(examples,
                                                            pad_id=self.pad_id,
                                                            mask_id=self.mask_id)

        batch = {"input_ids": inputs}
        batch['decoder_input_ids'] = decoder_inputs

        if self.mask_pad:
            # Encoder Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

            # Decoder Attention Mask, equivalent to key_padding_mask
            # Causal Mask is used in huggingface EncoderDecoder Model by default
            decoder_attention_mask = torch.ones_like(decoder_inputs)
            decoder_attention_mask[decoder_inputs == self.pad_id] = 0
            batch['decoder_attention_mask'] = decoder_attention_mask

        labels[labels == self.pad_id] = -100
        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


class AccDataset(Dataset):
    def __init__(self, melody_token_path, acc_token_path, shuffle=True):

        with open(melody_token_path) as f:
            melody_phrases = json.load(f)

        with open(acc_token_path) as f:
            acc_phrases = json.load(f)

        if shuffle:
            idx = list(range(len(acc_phrases)))
            random.shuffle(idx)
            self.melody_phrases = [melody_phrases[i]['token_ids'] for i in idx]
            self.acc_phrases = [acc_phrases[i]['token_ids'] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        return self.melody_phrases[index], self.acc_phrases[index]

    def __len__(self):
        return len(self.acc_phrases)


class MaskedAccDataset(Dataset):

    def __init__(self, melody_token_path, full_score_token_path=None, shuffle=True):

        with open(melody_token_path) as f:
            melody_phrases = json.load(f)

        with open(full_score_token_path) as f:
            phrases = json.load(f)

        self.melody_tokens = [i['token_ids'] for i in melody_phrases]
        self.melody_tokens += self.melody_tokens  # double for different masking

        self.tokens = [i['token_ids'] for i in phrases]
        self.tokens += self.tokens

        self.mask_idx = [i['rand_mask_idx'] for i in phrases] + \
            [i['center_mask_idx'] for i in melody_phrases]

        if shuffle:
            idx = list(range(len(self.phrases)))
            random.shuffle(idx)
            self.melody_phrases = [self.melody_phrases[i] for i in idx]
            self.phrases = [self.phrases[i] for i in idx]
            self.mask_idx = [self.mask_idx[i] for i in idx]

        # Dynamic mask measures for next phrase generation


def concat_measures(phrase, bar_id=89, eos_id=1):
    token_ids = deepcopy(phrase['ts-tp'])
    for ids in phrase['token_ids']:
        token_ids += ids
        token_ids += [bar_id]
    token_ids += [eos_id]
    return token_ids


def mask_concat_measures(phrase, bar_id=89, eos_id=1, mask_id=4, mask_ratio=.5):
    n_measure = len(phrase['token_ids'])
    n_mask_measure = int(n_measure * mask_ratio)
    i_mask_measure = sorted(random.sample(range(n_measure),
                                          n_mask_measure))

    token_ids = deepcopy(phrase['ts-tp'])
    for i, ids in enumerate(phrase['token_ids']):
        if i in i_mask_measure:
            token_ids += [mask_id for _ in ids]
        else:
            token_ids += ids
        token_ids += [bar_id]

    token_ids += [eos_id]
    return token_ids


def corrupt(tokens, vocab_candidate, corrupt_ratio):

    pos_orig = [(i, v) for i, v in enumerate(tokens) if v in vocab_candidate]
    n_corrupt = int(len(pos_orig) * corrupt_ratio)
    pos_orig = random.sample(pos_orig, n_corrupt)

    for pos, v in pos_orig:
        candidate = vocab_candidate.difference([v])
        tokens[pos] = random.choice(list(candidate))

    return tokens


def _mask_pair_collate_batch(
        examples, bar_id=89, pad_id=0, eos_id=1, mask_id=4, mask_ratio=.5,
        vocab_size=274, corrupt_ratio=.05, pred_masked_only=True):

    vocab_candidate = set(range(vocab_size)).difference([pad_id, eos_id, mask_id])

    encoder_inputs, decoder_inputs, labels = [], [], []
    for i, e in enumerate(examples):
        encoder_inputs.append(concat_measures(e[0], bar_id, eos_id))
        masked_decoder_input = mask_concat_measures(deepcopy(e[1]),
                                                    bar_id, eos_id, mask_id,
                                                    mask_ratio)
        if corrupt_ratio > 0:
            corrupt(masked_decoder_input, vocab_candidate, corrupt_ratio)

        decoder_inputs.append(masked_decoder_input)
        if len(e) < 3:
            labels.append(concat_measures(e[1], bar_id, eos_id))
        else:
            labels.append(concat_measures(e[2], bar_id, eos_id))

    if isinstance(encoder_inputs[0], (list, tuple, np.ndarray)):
        encoder_inputs = [torch.tensor(e, dtype=torch.long) for e in encoder_inputs]
        decoder_inputs = [torch.tensor(e, dtype=torch.long) for e in decoder_inputs]
        labels = [torch.tensor(e, dtype=torch.long) for e in labels]

    max_len_encoder = max(int(x.size(0)) for x in encoder_inputs)
    max_len_decoder = max(int(x.size(0)) for x in decoder_inputs)
    max_len = max(max_len_encoder, max_len_decoder)

    encoder_ids = encoder_inputs[0].new_full([len(examples), max_len], pad_id)
    decoder_ids = decoder_inputs[0].new_full([len(examples), max_len], pad_id)
    label_ids = decoder_inputs[0].new_full([len(examples), max_len], pad_id)

    for i in range(len(encoder_inputs)):
        encoder_seq_len = encoder_inputs[i].shape[0]
        encoder_ids[i, :encoder_seq_len] = encoder_inputs[i]

        decoder_seq_len = decoder_inputs[i].shape[0]
        decoder_ids[i, :decoder_seq_len] = decoder_inputs[i]

        if pred_masked_only:
            mask_idx = torch.where(decoder_ids[i] == mask_id)
            label_ids[i, mask_idx[0]] = labels[i][mask_idx]
        else:
            label_seq_len = labels[i].shape[0]
            label_ids[i, :label_seq_len] = labels[i]

    pad = torch.tensor([[mask_id] for _ in range(len(labels))], dtype=torch.long)
    decoder_ids = torch.cat((pad, decoder_ids[:, :-1]), 1)

    return encoder_ids, decoder_ids, label_ids


class MaskNextPhraseCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=0, mask_id=4, bar_id=89, mask_ratio=.5,
                 vocab_size=274, corrupt_ratio=.02,
                 mask_pad=True, pred_masked_only=True):

        self.pad_id = pad_id
        self.mask_id = mask_id
        self.bar_id = bar_id
        self.mask_pad = mask_pad
        self.mask_ratio = mask_ratio
        self.vocab_size = vocab_size
        self.corrupt_ratio = corrupt_ratio
        self.pred_masked_only = pred_masked_only

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        inputs, decoder_inputs, labels = _mask_pair_collate_batch(
            examples, pad_id=self.pad_id, mask_id=self.mask_id, bar_id=self.bar_id,
            mask_ratio=self.mask_ratio, vocab_size=self.vocab_size,
            corrupt_ratio=self.corrupt_ratio, pred_masked_only=self.pred_masked_only)

        batch = {"input_ids": inputs}
        batch['decoder_input_ids'] = decoder_inputs

        if self.mask_pad:
            # Encoder Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

            # Decoder Attention Mask, equivalent to key_padding_mask
            # Causal Mask is used in huggingface EncoderDecoder Model by default
            decoder_attention_mask = torch.ones_like(decoder_inputs)
            decoder_attention_mask[decoder_inputs == self.pad_id] = 0
            batch['decoder_attention_mask'] = decoder_attention_mask

        labels[labels == self.pad_id] = -100

        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


class MaskedSegmentDataset(Dataset):
    # For pretraining nsp
    def __init__(self, token_path=None, shuffle=True):

        with open(token_path) as f:
            segments = json.load(f)

        if shuffle:
            idx = list(range(len(segments)))
            random.shuffle(idx)
            self.segments = [segments[i] for i in idx]

    def __getitem__(self, index):
        """Return

        Args:
            index (int): index of entry
        """
        segment = self.segments[index]
        return segment

    def __len__(self):
        return len(self.segments)


class MaskedSegmentCollator():
    """
    Data collator
    """

    def __init__(self, pad_id=0, mask_id=4, mask_ratio=.5,
                 vocab_size=274, corrupt_ratio=.02,
                 mask_pad=True, pred_masked_only=True):

        self.pad_id = pad_id
        self.mask_id = mask_id
        self.mask_pad = mask_pad
        self.mask_ratio = mask_ratio
        self.vocab_size = vocab_size
        self.corrupt_ratio = corrupt_ratio
        self.pred_masked_only = pred_masked_only

    def __post_init__(self):
        pass

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.

        inputs, decoder_inputs, labels = _mask_seg_collate_batch(
            examples, pad_id=self.pad_id, mask_id=self.mask_id,
            mask_ratio=self.mask_ratio, vocab_size=self.vocab_size,
            corrupt_ratio=self.corrupt_ratio, pred_masked_only=self.pred_masked_only)

        batch = {"input_ids": inputs}
        batch['decoder_input_ids'] = decoder_inputs

        if self.mask_pad:
            # Encoder Attention Mask, equivalent to key_padding_mask
            attention_mask = torch.ones_like(inputs)
            attention_mask[inputs == self.pad_id] = 0
            batch['attention_mask'] = attention_mask

            # Decoder Attention Mask, equivalent to key_padding_mask
            # Causal Mask is used in huggingface EncoderDecoder Model by default
            decoder_attention_mask = torch.ones_like(decoder_inputs)
            decoder_attention_mask[decoder_inputs == self.pad_id] = 0
            batch['decoder_attention_mask'] = decoder_attention_mask

        labels[labels == self.pad_id] = -100

        # nn.CrossEntropy ignore -100 by default
        batch["labels"] = labels

        return batch


def mask_seg(tokens, mask_id, mask_ratio=.5):
    n_token = len(tokens)
    n_mask = int(n_token * mask_ratio)
    mask_idx = random.sample(list(range(n_token)), n_mask)
    for i in mask_idx:
        tokens[i] = mask_id
    return tokens


def _mask_seg_collate_batch(
        examples, pad_id=0, eos_id=1, mask_id=4, mask_ratio=.5, vocab_size=274,
        corrupt_ratio=.02, pred_masked_only=True):

    vocab_candidate = set(range(vocab_size)).difference([pad_id,
                                                         eos_id,
                                                         mask_id])

    encoder_inputs, decoder_inputs, labels = [], [], []
    for i, e in enumerate(examples):
        encoder_inputs.append(deepcopy(e))
        masked_decoder_input = mask_seg(deepcopy(e), mask_id, mask_ratio)
        if corrupt_ratio > 0:
            corrupt(masked_decoder_input, vocab_candidate, corrupt_ratio)
        decoder_inputs.append(masked_decoder_input)
        labels.append(e)

    if isinstance(encoder_inputs[0], (list, tuple, np.ndarray)):
        encoder_inputs = [torch.tensor(e, dtype=torch.long)
                          for e in encoder_inputs]
        decoder_inputs = [torch.tensor(e, dtype=torch.long)
                          for e in decoder_inputs]
        labels = [torch.tensor(e, dtype=torch.long) for e in labels]

    max_len_encoder = max(int(x.size(0)) for x in encoder_inputs)
    max_len_decoder = max(int(x.size(0)) for x in decoder_inputs)
    max_len = max(max_len_encoder, max_len_decoder)

    encoder_ids = encoder_inputs[0].new_full([len(examples), max_len], pad_id)
    decoder_ids = decoder_inputs[0].new_full([len(examples), max_len], pad_id)
    label_ids = decoder_inputs[0].new_full([len(examples), max_len], pad_id)

    for i in range(len(encoder_inputs)):
        encoder_seq_len = encoder_inputs[i].shape[0]
        encoder_ids[i, :encoder_seq_len] = encoder_inputs[i]

        decoder_seq_len = decoder_inputs[i].shape[0]
        decoder_ids[i, :decoder_seq_len] = decoder_inputs[i]

        if pred_masked_only:
            mask_idx = torch.where(decoder_ids[i] == mask_id)
            label_ids[i, mask_idx[0]] = labels[i][mask_idx]
        else:
            label_seq_len = labels[i].shape[0]
            label_ids[i, :label_seq_len] = labels[i]

    pad = torch.tensor([[mask_id] for _ in range(len(labels))], dtype=torch.long)
    decoder_ids = torch.cat((pad, decoder_ids[:, :-1]), 1)

    return encoder_ids, decoder_ids, label_ids
