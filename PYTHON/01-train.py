from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import torch
import os

def train_model():
    # Load processed dataset
    dataset = load_from_disk('ANALYSIS/processed_data')
    
    # Use a smaller model better suited for local training
    model_name = "facebook/opt-125m"  # Much smaller model (125M parameters)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        use_cache=False  # Disable KV-cache for training
    )
    
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256  # Reduced from 512 to save memory
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Take a subset of the data for testing
    tokenized_dataset = tokenized_dataset.select(range(min(1000, len(tokenized_dataset))))
    
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="ANALYSIS/results",
        num_train_epochs=1,             # Reduced epochs
        per_device_train_batch_size=1,  # Minimum batch size
        gradient_accumulation_steps=4,
        warmup_steps=50,
        logging_dir="ANALYSIS/logs",
        logging_steps=10,
        save_strategy="no",             # Don't save checkpoints
        learning_rate=1e-5,
        fp16=False,
        optim="adamw_torch",
        report_to="none",               # Disable wandb logging
        remove_unused_columns=True,
        no_cuda=True,                   # Force CPU training
    )
    
    print("Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("ANALYSIS/final_model")
    tokenizer.save_pretrained("ANALYSIS/final_model")
    print("Training complete!")

if __name__ == "__main__":
    # Set environment variables to control memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    train_model()
