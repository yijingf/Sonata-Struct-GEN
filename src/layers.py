import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding



def positionalencoding1d(n_embd, seq_len):
    """
    :param n_embd: dimension of the model
    :param seq_len: seq_len of positions
    :return: seq_len*n_embd position matrix
    """
    if n_embd % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(n_embd))
    pe = torch.zeros(seq_len, n_embd)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, n_embd, 2, dtype=torch.float) *
                         -(math.log(10000.0) / n_embd)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class MLPBlock(nn.Module):
    def __init__(self, n_embd, n_inter, dropout=.1):
        super().__init__()
        # hugging face use conv1d
        self.fc = nn.Linear(n_embd, n_inter, bias=False)
        self.proj = nn.Linear(n_inter, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)

        x = self.proj(x)
        return self.dropout(x)


class AttnBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_fc=3072, max_len=1024,
                 pe_type=None, attn_type='global', dropout=.1, block_len=128,
                 residual=False):

        super().__init__()

        self.residual = residual
        if residual:
            self.fc = nn.Linear(n_embd, n_embd, bias=False)
        else:
            self.fc = None

        self.attn_type = attn_type
        if attn_type == 'local':
            self.attn = LocalAttn(block_len)
        elif attn_type in ['global', 'perceiver-self']:
            self.attn = GlobalAttn(
                n_embd, n_head, max_len=max_len, dropout=dropout, pe_type=pe_type)
        elif attn_type == 'perceiver-cross':
            self.attn = PerceiverAttn(
                n_embd, n_head, max_len=max_len, dropout=dropout, pe_type=pe_type)
        else:
            raise ValueError("Unknown type of attention Module")

        self.ln_1 = nn.LayerNorm(n_embd)
        self.resid_dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp_block = MLPBlock(n_embd, n_fc, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, q_len=512):

        x = self.ln_1(x)
        if self.attn_type == 'perceiver-self':
            attn = self.attn(x, attn_mask, key_padding_mask, q_len)
        else:
            attn = self.attn(x, attn_mask, key_padding_mask)

        if self.attn_type == 'perceiver-cross':
            x = self.resid_dropout(x[:, -q_len:, :] + attn)
        else:
            x = self.resid_dropout(x + attn)

        # Implement residual structure for perceiver block
        if self.residual:
            x = self.fc(self.ln_1(x)) + x

        x = self.ln_2(x)
        x = self.mlp_block(x)

        return self.dropout(x)


class PerceiverAttn(nn.Module):
    def __init__(self, n_embd, n_head, max_len=1024, dropout=.1, pe_type='rotary'):
        super().__init__()
        d_head, remainder = divmod(n_embd, n_head)
        if remainder:
            raise ValueError("incompatible `n_embd` and `n_head`")

        self.max_len = max_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)

        self.pe_type = pe_type
        self.rotary_embd = None
        if self.pe_type == 'rotary':
            self.rotary_embd = RotaryEmbedding(d_head)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, attn_mask=None, key_padding_mask=None, q_len=512):
        """_summary_

        Args:
            x (_type_): _description_
            attn_mask (_type_, optional): It's already handled in forward, but still required by huggingface. Defaults to None.
            key_padding_mask (_type_, optional): _description_. Defaults to None.
            q_len (int, optional): _description_. Defaults to 512.

        Returns:
            _type_: _description_
        """
        batch_size, seq_len, _ = x.shape

        if q_len > seq_len:
            q_len = seq_len
        # assert q_len < seq_len, "Error: q_len >= seq_len"

        k = self.key(x).reshape(batch_size, seq_len,
                                self.n_head, -1).transpose(1, 2)
        # k_t.shape = (batch_size, n_head, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.n_head, -1).transpose(1, 2)
        # shape = (batch_size, n_head, seq_len, d_head)
        q = self.query(x[:, -q_len:, :]).reshape(batch_size, q_len,
                                                 self.n_head, -1).transpose(1, 2)
        # shape = (batch_size, n_head, q_len, d_head)

        if self.pe_type == 'rotary':
            q = self.rotary_embd.rotate_queries_or_keys(q)
            k = self.rotary_embd.rotate_queries_or_keys(k)

        QK_t = torch.matmul(q, k.transpose(2, 3))
        attn = QK_t / math.sqrt(q.size(-1))

        # q_len could be changed during inference. We don't register mask at initialization.
        mask = F.pad(torch.tril(torch.ones(q_len, q_len + 1)),
                     (seq_len - q_len - 1, 0), value=1).to(attn.device)
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0,
                                    float('-inf'))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, q_len, -1)
        return out


class GlobalAttn(nn.Module):
    """Original author: Jake Tae
    Modified: Yijing Feng (Add mask to QEr, add RoPE)
    """

    def __init__(self, n_embd, n_head, max_len=1024, dropout=0.1, pe_type='rel', d_rotary=32):
        super().__init__()
        d_head, remainder = divmod(n_embd, n_head)
        if remainder:
            raise ValueError("incompatible `n_embd` and `n_head`")

        self.max_len = max_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)

        self.pe_type = pe_type
        self.rotary_embd = None
        self.Er = None

        if self.pe_type == 'rel':
            # Relative Attention
            # Er can be shared across heads
            self.Er = nn.Parameter(torch.randn(max_len, d_head))
        elif self.pe_type == 'rotary':
            self.rotary_embd = RotaryEmbedding(d_rotary)
        elif self.pe_type == None:
            pass
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}.")

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x.shape == (batch_size, seq_len, n_embd)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        # 0: batch_size, 1: seq_len, 2: n_head, 3: d_head
        # k_t = self.key(x).reshape(batch_size, seq_len,
        #                           self.n_head, -1).permute(0, 2, 3, 1)
        k = self.key(x).reshape(batch_size, seq_len,
                                self.n_head, -1).transpose(1, 2)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.n_head, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.n_head, -1).transpose(1, 2)
        # shape = (batch_size, n_head, seq_len, d_head)

        if self.pe_type == 'rotary':
            q = self.rotary_embd.rotate_queries_or_keys(q)
            k = self.rotary_embd.rotate_queries_or_keys(k)

        QK_t = torch.matmul(q, k.transpose(2, 3))
        # k.transpose.shape = (batch_size, n_head, d_head, seq_len)
        # QK_t.shape = (batch_size, n_head, seq_len, seq_len)

        if self.pe_type == 'rel':
            start = self.max_len - seq_len
            Er_t = self.Er[start:, :].transpose(0, 1)
            # Er_t.shape = (d_head, seq_len)
            QEr = torch.matmul(q, Er_t)
            # QEr.shape = (batch_size, n_head, seq_len, seq_len)
            Srel = self._skew(QEr)
            # Srel.shape = (batch_size, n_head, seq_len, seq_len)
            attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        else:
            attn = QK_t / math.sqrt(q.size(-1))

        # Attention Mask
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, n_head, seq_len, seq_len)

        # Padding Mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0,
                                    float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, n_head, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, n_head, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, n_embd)
        return self.dropout(out)

    def _mask_QEr(sel, QEr):
        _, _, seq_len, _ = QEr.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).flip(1).to(QEr.device)
        return QEr.masked_fill(mask == 0, 0)

    def _skew(self, QEr):
        # Mask upper left triangle
        QEr = self._mask_QEr(QEr)

        # QEr.shape = (batch_size, n_head, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, n_head, seq_len, 1 + seq_len)
        batch_size, n_head, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, n_head, num_cols, num_rows)
        # reshaped.size = (batch_size, n_head, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, n_head, seq_len, seq_len)
        return Srel


class LocalAttn(nn.Module):

    def __init__(self, n_embd, n_head, block_len=128, dropout=0.1):

        d_head, remainder = divmod(n_embd, n_head)
        if remainder:
            raise ValueError(
                "incompatible `n_embd` and `n_head`"
            )

        self.block_len = block_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(2 * block_len - 1, d_head))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_len, block_len))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        # x.shape == (batch_size, seq_len, n_embd)
        batch_size, seq_len, _ = x.shape

        # If (seq_len < 2 * block_len), then we use only one block.
        if seq_len < 2 * self.block_len:
            block_len = seq_len
        else:
            block_len = self.block_len

        pad_len = torch.fmod(torch.tensor(-seq_len), torch.tensor(block_len))
        padded_x = F.pad(x, [0, 0, 0, pad_len, 0, 0])
        n_block = (seq_len + pad_len) / block_len
        padded_x = padded_x.reshape(int(batch_size * n_block), block_len, -1)

        k_t = self.key(padded_x).reshape(batch_size, block_len,
                                         self.n_head, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, n_head, d_head, seq_len)
        v = self.value(padded_x).reshape(batch_size, block_len,
                                         self.n_head, -1).transpose(1, 2)
        q = self.query(padded_x).reshape(batch_size, block_len,
                                         self.n_head, -1).transpose(1, 2)

        QK_t = torch.matmul(q, k_t)

        # Todo: attend to previous block, debug!
        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self._skew(QEr)

        attn = (QK_t + Srel) / math.sqrt(q.size(-1))

        # Todo: masking

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, n_head, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, n_head, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, n_embd)
        return self.dropout(out)

    def _mask_QEr(self, QEr):
        _, _, block_len, _ = QEr.shape
        mask_0 = torch.triu(torch.ones(block_len, 2 * block_len - 1)).flip(1)
        mask_1 = torch.triu(torch.ones(block_len, 2 * block_len - 1)).flip(0)
        mask = mask_0 * mask_1
        return QEr.masked_fill(mask == 0, 0)

    def _skew(self, QEr):
        # Mask upper left triangle
        batch_size, n_head, block_len, _ = QEr.shape
        QEr = self._mask_QEr(QEr)

        # QEr.shape = (batch_size, n_head, seq_len, seq_len)
        padded = F.pad(QEr, (0, 1))
        # padded.shape = (batch_size, n_head, seq_len, 1 + seq_len)

        # Flatten, append block_len - 1 0s
        padded = torch.concat([padded.flatten(), torch.zeros(block_len - 1)])

        reshaped = padded.reshape(
            batch_size, n_head, block_len + 1, 2 * block_len - 1)

        # reshaped.size = (batch_size, n_head, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, :block_len, -block_len:]
        # Srel.shape = (batch_size, n_head, seq_len, seq_len)
        return Srel
