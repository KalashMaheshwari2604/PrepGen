import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

def finetune_summarization_model():
    # --- 1. Define Model and Dataset Names ---
    # We're using a small, manageable model and a standard summarization dataset.
    model_name = "t5-small"
    dataset_name = "knkarthick/samsum"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. Load Dataset, Tokenizer, and Model ---
    print("Loading dataset, tokenizer, and model...")
    dataset = load_dataset(dataset_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # --- 3. Preprocessing Function ---
    # This function converts the text data into a format the T5 model understands (token IDs).
    def preprocess_function(examples):
        # The 'dialogue' is our input, and the 'summary' is our target output.
        # We add a "summarize: " prefix as T5 is an instruction-tuned model.
        inputs = ["summarize: " + doc for doc in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

        # The 'labels' are the tokenized summaries.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # --- 4. Apply Preprocessing to the Dataset ---
    print("Preprocessing the dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # --- 5. Set Up Training Arguments ---
    # These arguments control the fine-tuning process.
    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5-samsum-model",          # Where the trained model will be saved.
        num_train_epochs=3,                     # Number of times to go through the data.
        per_device_train_batch_size=4,          # Batch size. Crucial for fitting on a 4GB GPU.
        per_device_eval_batch_size=4,
        warmup_steps=500,                       # Number of steps to warm up the learning rate.
        weight_decay=0.01,                      # Regularization.
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",            # Evaluate during training.
        eval_steps=500,
        save_steps=500,                         # Save a checkpoint every 500 steps.
        load_best_model_at_end=True,
    )

    # Data collator handles creating batches of data during training.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # --- 6. Create the Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 7. Start Fine-Tuning ---
    print("Starting the fine-tuning process...")
    trainer.train(resume_from_checkpoint=True)

    # --- 8. Save the Final Model ---
    print("Training complete. Saving the final model.")
    trainer.save_model("./t5-samsum-model/final")

if __name__ == "__main__":
    finetune_summarization_model()