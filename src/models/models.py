"""Pytorch implementation of Music Transformer with Global/Local Relative Encoding
"""

import math
import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from layers import AttnBlock, positionalencoding1d
from config_utils import PerceiverARConfig, MusicTransformerConfig


class PerceiverARPretrained(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PerceiverARConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["AttnBlock"]

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PerceiverAR):
            module.gradient_checkpointing = value


class PerceiverAR(PerceiverARPretrained):
    def __init__(self, config):
        # (vocab_size, seq_len=None, n_embd=768, n_head=12, n_layer=12, dropout=.1, pe_type='rotary')
        super().__init__(config)

        self.n_layer = config.n_layer

        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Todo: positional encoding?

        # Cross Attn Layer
        self.cross_attn = AttnBlock(config.n_embd, config.n_head,
                                    max_len=config.n_positions,
                                    attn_type='perceiver-cross',
                                    pe_type=config.pe_type,
                                    dropout=config.pdrop,
                                    residual=config.residual)
        # Self Attn Layers
        self.self_attn = [AttnBlock(config.n_embd, config.n_head,
                                    max_len=config.n_positions,
                                    attn_type='perceiver-self',
                                    pe_type=config.pe_type,
                                    dropout=config.pdrop,
                                    residual=config.residual)
                          for _ in range(config.n_layer - 1)]

        self.layers = nn.ModuleList([self.cross_attn] + self.self_attn)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model Parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.n_layer),
                           range(torch.cuda.device_count()))  # Todo: double check n_layer
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, self.n_layer)

        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + \
            str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        self.wte = self.wte.to(self.first_device)
        # self.wpe = self.wpe.to(self.first_device)

        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.layers[block] = self.layers[block].to(cuda_device)

        # ln_f to last device
        self.ln_f = self.ln_f.to(self.last_device)

        self.lm_head = self.lm_head.to(self.first_device)
        self.model_parallel = True
        return

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        # self.wpe = self.wpe.to("cpu")

        self.cross_attn = self.cross_attn.to("cpu")

        for index in range(len(self.n_layer)):
            self.layers[index] = self.layers[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")

        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(self, input_ids, q_len=512, labels=None, return_dict=True, **kwargs):
        x = self.wte(input_ids) / math.sqrt(self.n_embd)
        # Todo: Globa PE?

        for layer in self.layers:
            if layer.attn_type == 'perceiver-cross':
                x = layer(x, q_len)
            else:
                x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss()  # -100 index = padding token
            # Flatten the tokens
            loss = loss_fn(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return CausalLMOutput(loss=loss, logits=logits)


class MusicTransformerPretrained(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MusicTransformerConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["AttnBlock"]

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PerceiverAR):
            module.gradient_checkpointing = value


class MusicTransformer(MusicTransformerPretrained):
    """No difference from vanila transformer except for relative PE for each attn block implemented in an efficient way
    Args:
        MusicTransformerPretrained (_type_): _description_
    """

    def __init__(self, config):
        # (vocab_size, seq_len=None, n_embd=768, n_head=12, n_layer=12, dropout=.1, pe_type='rotary')
        # def __init__(self, vocab_size, seq_len=None, n_embd=768, n_head=12, n_layer=12, dropout=.1, pe_type='rel'):
        super().__init__(config)

        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Todo: positional encoding
        # Is it really necessary since we have rel-pe for each attn block?

        self.layers = nn.ModuleList([AttnBlock(config.n_embd, config.n_head,
                                               max_len=config.n_positions,
                                               n_fc=config.n_fc,
                                               attn_type='global',
                                               pe_type=config.pe_type,
                                               dropout=config.pdrop)
                                     for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.n_layer),
                           range(torch.cuda.device_count()))  # Todo: double check n_layer
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, self.n_layer)

        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + \
            str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        self.wte = self.wte.to(self.first_device)
        # self.wpe = self.wpe.to(self.first_device)

        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.layers[block] = self.layers[block].to(cuda_device)

        # ln_f to last device
        self.ln_f = self.ln_f.to(self.last_device)

        self.lm_head = self.lm_head.to(self.first_device)
        self.model_parallel = True
        return

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        # self.wpe = self.wpe.to("cpu")

        self.cross_attn = self.cross_attn.to("cpu")

        for index in range(len(self.n_layer)):
            self.layers[index] = self.layers[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")

        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(
            self, input_ids, labels=None, attn_mask=None, key_padding_mask=None,
            return_dict=True, **kwargs):
        """
        Args:
            input_ids (_type_): _description_
            labels (_type_, optional): _description_. Defaults to None.
            attn_mask (_type_, optional): Causal mask. Defaults to None.
            key_padding_mask (_type_, optional): _description_. Defaults to None.
            return_dict (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        x = self.wte(input_ids) / math.sqrt(self.n_embd)
        # PE
        # _, seq_len = input_ids.shape
        # pe = positionalencoding1d(self.n_embd, seq_len)
        # x = x + pe.to(x.device)

        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss()  # -100 index = padding token
            # Flatten the tokens
            loss = loss_fn(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)
