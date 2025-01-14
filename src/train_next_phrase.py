from transformers import Trainer
from transformers import TrainingArguments
from transformers import EncoderDecoderModel
from transformers import BertConfig, EncoderDecoderConfig


from kern_utils.tokenizer import BertTokenizer
# from utils.tokenizer import BertTokenizer
from data_loader import BasePairDataset, BaseNextPhraseCollator

max_len = 512
base_vocab_file = "../sonata-dataset-phrase/vocab/base_vocab_melody.txt"

with open(base_vocab_file) as f:
    base_vocab = f.read().splitlines()

tokenizer = BertTokenizer()
tokenizer.train(base_vocab)
vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_id
mask_id = tokenizer.mask_id

config_encoder = BertConfig(vocab_size=vocab_size,
                            num_hidden_layers=4,
                            hidden_size=768,
                            num_attention_heads=8,
                            intermediate_size=1024,
                            max_position_embeddings=max_len)

config_decoder = BertConfig(vocab_size=vocab_size,
                            num_hidden_layers=4,
                            hidden_size=768,
                            num_attention_heads=8,
                            intermediate_size=1024,
                            max_position_embeddings=max_len)

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder,
                                                           config_decoder)
config.pad_token_id = pad_id
model = EncoderDecoderModel(config)

train_token_path = "../sonata-dataset-phrase/dataset/melody/train_512_pad.json"
train_dataset = BasePairDataset(train_token_path)

eval_token_path = "../sonata-dataset-phrase/dataset/melody/val_512_pad.json"
eval_dataset = BasePairDataset(eval_token_path)

data_collator = BaseNextPhraseCollator(pad_id=pad_id)


model_output_dir = "../models/melody"
n_epochs = 50
batch_size = 32

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

trainer.train()
trainer.save_model()
