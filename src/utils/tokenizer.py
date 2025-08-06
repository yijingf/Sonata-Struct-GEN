import json
from utils.decode import decode_token_to_pm


class BaseTokenizer():

    def __init__(self):

        self.pad_token = "pad"
        self.eos_token = "eos"
        self.unk_token = "unk"

        self.special_tokens = ["pad", "eos", "unk"]

        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2

        self.special_token_ids = [0, 1, 2]
        self.n_special_token = 3

        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}

    def load_base_vocab(self, base_vocab_file):
        with open(base_vocab_file) as f:
            tokens = f.read().splitlines()

        self.train(tokens)
        return

    def load_vocab(self, vocab_file):
        with open(vocab_file) as f:
            tokens = f.read().splitlines()

        ids = list(range(len(tokens)))
        self.token_to_id = dict(zip(tokens, ids))
        self.id_to_token = dict(zip(ids, tokens))

        # Sanity Check
        for token in self.special_tokens:
            assert token not in self.token_to_id, f"Token {token} not in vocab."
            assert self.token_to_id[token] == getattr(self, f"{token}_id")

        assert tokens[self.n_special_token - 1] in self.special_tokens
        self.vocab = tokens
        self.vocab_size = len(tokens)
        return

    def has_irregular_token(self, tokens):
        return any([token not in self.vocab for token in tokens])

    def save_vocab(self, vocab_file):
        with open(vocab_file, "w") as f:
            for token in self.vocab:
                f.write(token + "\n")

    def load_vocab_json(self, vocab_json_file):
        with open(vocab_json_file) as f:
            self.token_to_id = json.load(f)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Sanity Check
        for token in self.special_tokens:
            assert token in self.token_to_id, f"Token {token} not in vocab."
            assert self.token_to_id[token] == getattr(self, f"{token}_id")

        self.vocab_size = len(self.id_to_token)
        self.vocab = [self.id_to_token[i] for i in range(self.vocab_size)]
        return

    def save_vocab_json(self, vocab_json_file):
        with open(vocab_json_file, "w") as f:
            json.dump(self.token_to_id, f)
        return

    def train(self, tokens):
        tokens = sorted(set(tokens))

        for token in self.special_tokens:
            if token in tokens:
                tokens.remove(token)

        self.vocab = self.special_tokens + tokens
        self.vocab_size = len(self.vocab)
        ids = list(range(self.vocab_size))

        self.token_to_id = dict(zip(self.vocab, ids))
        self.id_to_token = dict(zip(ids, self.vocab))

        return

    def add_special_token(self, token):
        if token in self.special_tokens:
            Warning(f"Token: {token} is already a special token.")
            return

        token_attr = f"{token}_token"
        setattr(self, token_attr, token)

        self.special_tokens += [token]

        id_attr = f"{token}_id"
        setattr(self, id_attr, self.n_special_token)
        self.special_token_ids += [self.n_special_token]
        self.n_special_token += 1

        return

    def convert_tokens_to_ids(self, tokens):
        token_ids = [self.token_to_id[i] for i in tokens]
        return token_ids

    def convert_ids_to_tokens(self, token_ids):
        tokens = [self.id_to_token[i] for i in token_ids]
        return tokens

    def postprocess(self, token_ids):

        if self.eos_id in token_ids:
            idx = token_ids.index(self.eos_id)
            return token_ids[:idx]
        return token_ids

        # # Trim padding or eos token
        # processed_token_ids = []
        # for token_id in token_ids:
        #     if token_id in [self.eos_id, self.pad_id]:
        #         break
        #     else:
        #         processed_token_ids.append(token_id)

        # return processed_token_ids


class MTTokenizer(BaseTokenizer):

    def __init__(self, vocab_file=None, vocab_json_file=None):

        super().__init__()

        if vocab_file:
            self.load_vocab(vocab_file)

        if vocab_json_file:
            self.load_vocab_json(vocab_json_file)

        return

    def decode(self, token_ids):

        token_ids = self.postprocess(token_ids)
        tokens = self.convert_ids_to_tokens(token_ids)

        pm = decode_token_to_pm(tokens, bar_eos_token=None)

        return pm


class BertTokenizer(BaseTokenizer):

    def __init__(self, vocab_file=None, vocab_json_file=None, add_sep=True):

        super().__init__()
        if add_sep:
            self.add_special_token("sep")  # used as end of bar token
        else:
            self.sep_token = None
        self.add_special_token("mask")

        if vocab_file:
            self.load_vocab(vocab_file)

        if vocab_json_file:
            self.load_vocab_json(vocab_json_file)

        return

    def decode(self, token_ids, bar_eos_token=None):

        token_ids = self.postprocess(token_ids)
        tokens = self.convert_ids_to_tokens(token_ids)

        if not bar_eos_token and self.sep_token:
            bar_eos_token = self.sep_token
        pm = decode_token_to_pm(tokens, bar_eos_token)

        return pm
