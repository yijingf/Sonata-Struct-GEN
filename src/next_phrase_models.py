import torch
import torch.nn as nn
from transformers import EncoderDecoderConfig

from typing import Optional, Tuple, Union
from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoModelForCausalLM

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class NextPhraseModel(PreTrainedModel):

    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)

        self.encoder = AutoModel.from_config(config.encoder)
        self.decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs,
                ) -> Union[Tuple, Seq2SeqLMOutput]:

        hidden_0 = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state
        hidden_1 = self.encoder(input_ids=decoder_input_ids,
                                attention_mask=decoder_attention_mask).last_hidden_state

        out = self.decoder(encoder_hidden_states=hidden_0,
                           inputs_embeds=hidden_1,
                           labels=labels)

        return out
