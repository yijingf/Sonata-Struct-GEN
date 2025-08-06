import os
from transformers import Trainer
from transformers import TrainingArguments
from transformers import EncoderDecoderModel
from transformers import BertConfig, EncoderDecoderConfig

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tokenizer import BertTokenizer
from utils.data_loader import BasePairDataset, BaseNextPhraseCollator
from utils.constants import DATA_DIR, MODEL_DIR


# Constants
model_name = "nextGEN"
max_len = 512
pretrain_dataset_dir = os.path.join(DATA_DIR, "dataset", f"{model_name}_pretrain")
finetune_dataset_dir = os.path.join(DATA_DIR, "dataset", f"{model_name}_finetune")
model_dir = os.path.join(MODEL_DIR, model_name)

# Tokenizer
base_vocab_file = os.path.join(DATA_DIR, "vocab", "base_vocab.txt")
tokenizer = BertTokenizer()
tokenizer.load_base_vocab(base_vocab_file)


def main(n_epochs, batch_size, pretrain=False):

    config_encoder = BertConfig(vocab_size=tokenizer.vocab_size,
                                num_hidden_layers=4,
                                hidden_size=768,
                                num_attention_heads=8,
                                intermediate_size=1024,
                                max_position_embeddings=max_len)

    config_decoder = BertConfig(vocab_size=tokenizer.vocab_size,
                                num_hidden_layers=4,
                                hidden_size=768,
                                num_attention_heads=8,
                                intermediate_size=1024,
                                max_position_embeddings=max_len)

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder,
                                                               config_decoder)
    config.pad_token_id = tokenizer.pad_id
    model = EncoderDecoderModel(config)

    if pretrain:
        train_token_path = os.path.join(pretrain_dataset_dir, "train_512_pad.json")
        eval_token_path = os.path.join(pretrain_dataset_dir, "val_512_pad.json")
    else:
        train_token_path = os.path.join(finetune_dataset_dir, "train_512_pad.json")
        eval_token_path = os.path.join(finetune_dataset_dir, "val_512_pad.json")

    train_dataset = BasePairDataset(train_token_path)
    eval_dataset = BasePairDataset(eval_token_path)
    data_collator = BaseNextPhraseCollator(pad_id=tokenizer.pad_id)

    training_args = TrainingArguments(
        output_dir=model_dir,  # The output directory
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

    if pretrain:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=model_dir)
    trainer.save_model()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Train epoch. Defaults to 50")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per device. Defaults to 32.")
    parser.add_argument("--pretrain", action="store_true",
                        help="Pretrain.")
    args = parser.parse_args()

    main(args.n_epochs, args.batch_size, args.pretrain)
