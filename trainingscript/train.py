from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

from transformers import TextDataset

def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
    return dataset

train_dataset = load_dataset("train.txt", tokenizer)
valid_dataset = load_dataset("valid.txt", tokenizer)

from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

