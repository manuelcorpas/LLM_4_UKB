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

os.makedirs("RESULTS/opt-1.3b/offload", exist_ok=True)

def train_model():
    # Load processed dataset
    dataset = load_from_disk('RESULTS/opt-1.3b/processed_data')
    
    # Use a smaller model better suited for local training
    model_name = "facebook/opt-1.3b"  # Much smaller model (125M parameters)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        use_cache=False,  # Disable KV-cache for training
        offload_folder="RESULTS/opt-1.3b/offload"  # Add this line
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
        output_dir="RESULTS/opt-1.3b",
        num_train_epochs=1,             
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=16,
        warmup_steps=100,
        logging_dir="RESULTS/opt-1.3b/logs",
        logging_steps=10,
        save_strategy="epoch",             
        learning_rate=5e-6,
        fp16=False,  # Changed this to False since we're using CPU
        optim="adamw_torch",
        report_to="none",               
        remove_unused_columns=True,
        use_cpu=True,  # Changed from no_cuda=True to use_cpu=True
        gradient_checkpointing=True,
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
    trainer.save_model("RESULTS/opt-1.3b/final_model")
    tokenizer.save_pretrained("RESULTS/opt-1.3b/final_model")
    print("Training complete!")

if __name__ == "__main__":
    # Set environment variables to control memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    train_model()
