import os
from transformers import Trainer
from transformers import TrainingArguments
from data_loader import MaskedPairDataset, MaskNextPhraseCollator


def build_model(vocab_size, max_len=512, pad_id=0):

    from transformers import EncoderDecoderModel, EncoderDecoderConfig
    from transformers import RoFormerModel, RoFormerForCausalLM, RoFormerConfig

    config_encoder = RoFormerConfig(vocab_size=vocab_size,
                                    num_hidden_layers=4,
                                    hidden_size=768,
                                    num_attention_heads=8,
                                    intermediate_size=1024,
                                    max_position_embeddings=max_len,)

    config_decoder = RoFormerConfig(vocab_size=vocab_size,
                                    num_hidden_layers=4,
                                    hidden_size=768,
                                    num_attention_heads=8,
                                    intermediate_size=1024,
                                    max_position_embeddings=max_len,)

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder,
                                                               config_decoder)
    config.pad_token_id = pad_id
    model = EncoderDecoderModel(config,
                                RoFormerModel(config_encoder),
                                RoFormerForCausalLM(config_decoder))

    return model


def main(train_path, eval_path,
         src_data_type='krn', base_vocab_file=None,
         max_len=512, bar_pad=False,
         batch_size=32, n_epochs=100,
         vocab_size=500, pad_id=0, mask_id=None, mask_pad=True,
         pred_masked_only=True, mask_mode='mix',
         model_dir="../models", checkpoint_path=None):

    # Load Dataset
    if src_data_type == 'krn':

        from utils.tokenizer import BertTokenizer

        with open(base_vocab_file) as f:
            base_vocab = f.read().splitlines()
        tokenizer = BertTokenizer()
        tokenizer.train(base_vocab)
        vocab_size = tokenizer.vocab_size
        pad_id = tokenizer.pad_id
        mask_id = tokenizer.mask_id

    train_dataset = MaskedPairDataset(train_path, seq_len=max_len)
    eval_dataset = MaskedPairDataset(eval_path, seq_len=max_len)

    data_collator = MaskNextPhraseCollator(max_len=max_len,
                                           mask_pad=mask_pad)

    # Build model
    model = build_model(vocab_size,
                        max_len=max_len,
                        pad_id=pad_id)

    # Setup Training Args
    os.makedirs(model_dir, exist_ok=True)

    prefix = "next-phrase"

    if bar_pad:
        prefix = f"{prefix}-pad"

    if not pred_masked_only:
        prefix = f"{prefix}-all"

    model_output_dir = os.path.join(
        model_dir, f"{prefix}-{src_data_type}-{max_len}")

    training_args = TrainingArguments(
        output_dir=model_output_dir,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=n_epochs,  # number of training epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
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
