import pandas as pd
import os
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

os.environ["WANDB_DISABLED"] = "true"
# 1. Load your CSV dataset
csv_path = "dataset.csv"
df = pd.read_csv(csv_path)
df = df.dropna()

# Make sure the columns are named "en" and "fr"
df.columns = ["en", "fr"]

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 2. Load pretrained tokenizer and model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 3. Preprocessing function
def preprocess(example):
    inputs = tokenizer(example["en"], truncation=True, padding="max_length", max_length=64)
    targets = tokenizer(example["fr"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply preprocessing
tokenized_data = dataset.map(preprocess, batched=True)

# 4. Training setup
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt_output",
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs"
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 5. Train the model
trainer.train()

# 6. Save the trained model
trainer.save_model("./trained_translation_model")
tokenizer.save_pretrained("./trained_translation_model")

print("✅ Training complete. Model saved in './trained_translation_model'")
