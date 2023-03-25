import pandas as pd
from transformers import GPT2Tokenizer, TextDataset, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def load_dataset(file_path, tokenizer):
    data = pd.read_csv(file_path)
    data['combined'] = data['Item'] + ' | ' + data['Category']
    text = '\n'.join(data['combined'].tolist())

    with open('temp.txt', 'w') as f:
        f.write(text)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='temp.txt',
        block_size=128
    )
    return dataset

train_dataset = load_dataset("mylist.csv", tokenizer)
valid_dataset = load_dataset("mylist.csv", tokenizer)

config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")
