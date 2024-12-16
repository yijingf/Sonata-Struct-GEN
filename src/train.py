import os
from transformers import Trainer
from transformers import TrainingArguments
from data_loader import BaseDataset, BaseDataCollator


def build_model(vocab_size, model_type="mass", max_len=512, pad_id=0):

    if model_type == "music-transformer":
        from models import MusicTransformer, MusicTransformerConfig
        config = MusicTransformerConfig(vocab_size=vocab_size,
                                        n_positions=max_len,
                                        n_head=8,
                                        n_layer=8,
                                        n_embd=768,
                                        n_fc=1024,
                                        pdrop=.1,)
        model = MusicTransformer(config)

    elif model_type == "bert":
        from transformers import BertConfig, BertForMaskedLM
        config = BertConfig(vocab_size=vocab_size,
                            hidden_size=768,
                            num_hidden_layers=8,
                            num_attention_heads=8,
                            intermediate_size=1024,
                            max_position_embeddings=max_len,
                            position_embedding_type="relative_key_query")
        model = BertForMaskedLM(config)

    elif model_type == "roberta":
        from transformers import RobertaConfig, RobertaForMaskedLM
        config = RobertaConfig(
            vocab_size=vocab_size, hidden_size=768, num_hidden_layers=8,
            num_attention_heads=8, intermediate_size=1024,
            max_position_embeddings=max_len,
            position_embedding_type="relative_key_query")

        model = RobertaForMaskedLM(config)

    elif model_type == "mass" or model_type == "encoder-decoder":
        from transformers import EncoderDecoderModel
        from transformers import BertConfig, EncoderDecoderConfig

        config_encoder = BertConfig(vocab_size=vocab_size,
                                    num_hidden_layers=4,
                                    hidden_size=768,
                                    num_attention_heads=8,
                                    intermediate_size=1024,
                                    max_position_embeddings=max_len,)
        # position_embedding_type="relative_key_query")

        config_decoder = BertConfig(vocab_size=vocab_size,
                                    num_hidden_layers=4,
                                    hidden_size=768,
                                    num_attention_heads=8,
                                    intermediate_size=1024,
                                    max_position_embeddings=max_len,)
        # position_embedding_type="relative_key_query")

        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder,
                                                                   config_decoder)
        config.pad_token_id = pad_id
        model = EncoderDecoderModel(config)

    else:
        raise ValueError(f"Unknown pretrained model type: {model_type}")

    return model


def main(train_path, eval_path,
         src_data_type="krn", base_vocab_file=None,
         max_len=512, bar_pad=False,
         batch_size=32, n_epochs=100,
         vocab_size=500, pad_id=0, mask_id=None, mask_pad=True,
         model_type="mass", pred_masked_only=True, mask_mode="mix",
         model_dir="../models", checkpoint_path=None):

    # Load Dataset
    if src_data_type == "krn":

        from utils.tokenizer import BertTokenizer

        with open(base_vocab_file) as f:
            base_vocab = f.read().splitlines()
        tokenizer = BertTokenizer()
        tokenizer.train(base_vocab)
        vocab_size = tokenizer.vocab_size
        pad_id = tokenizer.pad_id
        mask_id = tokenizer.mask_id

    # Todo: MIDI Tokenizer

    if model_type not in [
            "bert", "roberta", "mass", "encoder-decoder"]:  # non-masking model
        train_dataset = BaseDataset(train_path, seq_len=max_len)
        eval_dataset = BaseDataset(eval_path, seq_len=max_len)
        data_collator = BaseDataCollator(max_len=max_len, mask_pad=mask_pad)

    elif model_type == "encoder-decoder":  # Todo: double check
        train_dataset = BaseDataset(train_path, seq_len=max_len)
        eval_dataset = BaseDataset(eval_path, seq_len=max_len)
        data_collator = BaseDataCollator(max_len=max_len, mask_pad=mask_pad)

    elif model_type == "bert":
        from data_loader import MaskedDataset, BertDataCollator
        train_dataset = MaskedDataset(train_path, seq_len=max_len)
        eval_dataset = MaskedDataset(eval_path, seq_len=max_len)
        data_collator = BertDataCollator(vocab_size=vocab_size,
                                         pad_id=pad_id,
                                         mask_id=mask_id,
                                         mask_pad=mask_pad)

    elif model_type == "roberta":
        from data_loader import MaskedDataset, BertDataCollator
        train_dataset = MaskedDataset(train_path, seq_len=max_len - 1)
        eval_dataset = MaskedDataset(eval_path, seq_len=max_len - 1)
        data_collator = BertDataCollator(vocab_size=vocab_size,
                                         pad_id=pad_id,
                                         mask_id=mask_id,
                                         mask_pad=mask_pad)

    elif model_type == "mass":
        from data_loader import MaskedDataset, MassDataCollator
        train_dataset = MaskedDataset(train_path, seq_len=max_len,
                                      mask_mode=mask_mode)
        eval_dataset = MaskedDataset(eval_path, seq_len=max_len,
                                     mask_mode=mask_mode)
        data_collator = MassDataCollator(mask_id=mask_id,
                                         pad_id=pad_id,
                                         pred_masked_only=pred_masked_only,
                                         mask_pad=mask_pad)

    else:
        raise ValueError("invalid model type")

    # Build model
    model = build_model(vocab_size,
                        model_type=model_type,
                        max_len=max_len,
                        pad_id=pad_id)

    # Setup Training Args
    os.makedirs(model_dir, exist_ok=True)

    prefix = f"{model_type}"

    if bar_pad:
        prefix = f"{prefix}-pad"

    if not pred_masked_only:
        prefix = f"{prefix}-all"

    if model_type == "mass" and mask_mode != "mix":
        prefix = f"{prefix}-{mask_mode}"

    model_output_dir = os.path.join(
        model_dir, f"{prefix}-{src_data_type}-{max_len}")

    training_args = TrainingArguments(
        output_dir=model_output_dir,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=n_epochs,  # number of training epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model()
    return


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()

    parser.add_argument("--arg_file", dest="arg_file", type=str,
                        default="", help="Arguments .json file.")
    input_args = parser.parse_args()

    if input_args.arg_file:
        with open(input_args.arg_file) as f:
            args = json.load(f)

    main(**args)
