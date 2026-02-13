# ============================================================================
# KAGGLE T5 FINE-TUNING - SCIENTIFIC PAPERS FOCUS
# Mixed Dataset: 70% Scientific + 20% BookSum + 10% WikiHow
# Copy each cell separately to Kaggle notebook
# ============================================================================

# ============================================================================
# CELL 1: Installation & Fixes (Run First, ~2 minutes)
# ============================================================================

print("📦 Installing packages and fixing compatibility issues...")

# Fix protobuf compatibility issue
!pip uninstall -y protobuf
!pip install -q protobuf==3.20.3

# Install required packages
!pip install -q transformers datasets accelerate sentencepiece evaluate rouge_score

print("\n✅ Installation complete!")
print("⚠️  Ignore any dependency warnings - they won't affect training")
print("🔄 If you see CUDA warnings above, that's normal - training will still work")

# ============================================================================
# CELL 2: Imports & Configuration (Run Second, ~10 seconds)
# ============================================================================

import os
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np
from tqdm.auto import tqdm
import time
import json
from collections import Counter

print("✅ All imports successful!")

# Configuration
CONFIG = {
    # Model settings
    "model_name": "google/flan-t5-base",
    "output_dir": "./academic-summarizer-scientific",
    
    # Mixed Dataset Strategy (70-20-10 split)
    "scientific_samples": 20000,  # 70% - arXiv scientific papers
    "booksum_samples": 6000,      # 20% - Long-form book summaries
    "wikihow_samples": 2500,      # 10% - Instructional articles
    "total_samples": 28500,       # Optimized for Kaggle 18.5GB
    
    # Training hyperparameters
    "learning_rate": 3e-4,
    "num_epochs": 3,
    "batch_size": 8,
    "gradient_accumulation": 4,
    
    # Text settings
    "max_input_length": 1024,
    "max_target_length": 512,
    
    # Optimization
    "fp16": True,
    "warmup_steps": 500,
}

print("\n🚀 Configuration:")
print("=" * 60)
print(f"Model: {CONFIG['model_name']}")
print(f"Dataset Mix: 70% Scientific + 20% BookSum + 10% WikiHow")
print(f"Total samples: {CONFIG['total_samples']:,}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ============================================================================
# CELL 3: Load Datasets (Run Third, ~5-10 minutes)
# ============================================================================

print("\n📥 Loading datasets (this takes 5-10 minutes)...")
print("💡 TIP: You can grab coffee while this runs!\n")

# Dataset 1: Scientific Papers (arXiv abstracts)
print("1️⃣ Loading Scientific Papers (arXiv)...")
scientific_dataset = load_dataset(
    "scientific_papers",
    "arxiv",
    split=f"train[:{CONFIG['scientific_samples']}]",
    trust_remote_code=True
)
print(f"   ✅ Loaded {len(scientific_dataset):,} scientific papers")

# Dataset 2: BookSum (Long-form summaries)
print("\n2️⃣ Loading BookSum...")
booksum_dataset = load_dataset(
    "kmfoda/booksum",
    split=f"train[:{CONFIG['booksum_samples']}]",
    trust_remote_code=True
)
print(f"   ✅ Loaded {len(booksum_dataset):,} book summaries")

# Dataset 3: WikiHow (Instructional content)
print("\n3️⃣ Loading WikiHow...")
wikihow_dataset = load_dataset(
    "wikihow",
    "all",
    split=f"train[:{CONFIG['wikihow_samples']}]",
    trust_remote_code=True
)
print(f"   ✅ Loaded {len(wikihow_dataset):,} WikiHow articles")

print("\n✅ All datasets loaded successfully!")

# ============================================================================
# CELL 4: Process & Combine Datasets (Run Fourth, ~2-3 minutes)
# ============================================================================

print("\n🔄 Processing and combining datasets...")

# Process Scientific Papers
def process_scientific(example):
    return {
        "document": example["article"],
        "summary": example["abstract"],
        "source": "scientific"
    }

scientific_processed = scientific_dataset.map(
    process_scientific,
    remove_columns=scientific_dataset.column_names,
    desc="Processing Scientific Papers"
)

# Process BookSum
def process_booksum(example):
    return {
        "document": example["chapter"],
        "summary": example["summary_text"],
        "source": "booksum"
    }

booksum_processed = booksum_dataset.map(
    process_booksum,
    remove_columns=booksum_dataset.column_names,
    desc="Processing BookSum"
)

# Process WikiHow
def process_wikihow(example):
    return {
        "document": example["text"],
        "summary": example["headline"],
        "source": "wikihow"
    }

wikihow_processed = wikihow_dataset.map(
    process_wikihow,
    remove_columns=wikihow_dataset.column_names,
    desc="Processing WikiHow"
)

# Combine datasets
combined_dataset = concatenate_datasets([
    scientific_processed,
    booksum_processed,
    wikihow_processed
])

# Shuffle and split
combined_dataset = combined_dataset.shuffle(seed=42)
dataset_split = combined_dataset.train_test_split(test_size=0.1, seed=42)

dataset_dict = DatasetDict({
    'train': dataset_split['train'],
    'validation': dataset_split['test']
})

# Show distribution
sources = dataset_dict['train']['source']
distribution = Counter(sources)

print("\n✅ Datasets combined!")
print(f"\n📊 Final Split:")
print(f"  Train: {len(dataset_dict['train']):,}")
print(f"  Validation: {len(dataset_dict['validation']):,}")
print(f"\n📈 Training Distribution:")
for source, count in distribution.items():
    percentage = (count / len(sources)) * 100
    print(f"  {source}: {count:,} ({percentage:.1f}%)")

# ============================================================================
# CELL 5: Load Model & Tokenizer (Run Fifth, ~1 minute)
# ============================================================================

print("\n🤖 Loading T5 model and tokenizer...")

tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"])
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])

print(f"✅ Model loaded!")
print(f"   Name: {CONFIG['model_name']}")
print(f"   Parameters: {model.num_parameters() / 1e6:.1f}M")

# ============================================================================
# CELL 6: Tokenize Datasets (Run Sixth, ~5-8 minutes)
# ============================================================================

print("\n🔄 Tokenizing datasets (this takes 5-8 minutes)...")

def preprocess_function(examples):
    # Task-specific prefixes
    prefixes = []
    for source in examples["source"]:
        if source == "scientific":
            prefixes.append("summarize scientific paper: ")
        elif source == "booksum":
            prefixes.append("summarize book chapter: ")
        else:  # wikihow
            prefixes.append("summarize instructions: ")
    
    inputs = [prefix + doc for prefix, doc in zip(prefixes, examples["document"])]
    targets = examples["summary"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_input_length"],
        truncation=True,
        padding="max_length",
    )
    
    labels = tokenizer(
        targets,
        max_length=CONFIG["max_target_length"],
        truncation=True,
        padding="max_length",
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset_dict.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_dict["train"].column_names,
    desc="Tokenizing"
)

print("✅ Tokenization complete!")

# ============================================================================
# CELL 7: Setup Training (Run Seventh, ~30 seconds)
# ============================================================================

print("\n⚙️ Setting up training configuration...")

# Metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "avg_length": result["gen_len"]
    }

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    weight_decay=0.01,
    fp16=CONFIG["fp16"],
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    logging_steps=100,
    logging_dir=f"{CONFIG['output_dir']}/logs",
    predict_with_generate=True,
    generation_max_length=CONFIG["max_target_length"],
    generation_num_beams=4,
    dataloader_num_workers=2,
    remove_unused_columns=True,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized!")

# ============================================================================
# CELL 8: START TRAINING (Run Eighth, ~5-6 HOURS)
# ============================================================================
# ⚠️ WARNING: This cell takes 5-6 hours!
# 💡 TIP: Leave it running overnight or during classes
# ============================================================================

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

# Calculate time estimates
total_samples = len(tokenized_dataset['train'])
effective_batch = CONFIG['batch_size'] * CONFIG['gradient_accumulation']
steps_per_epoch = total_samples // effective_batch
total_steps = steps_per_epoch * CONFIG['num_epochs']
time_per_step_minutes = 0.08
estimated_hours = (total_steps * time_per_step_minutes) / 60

print(f"Total samples: {total_samples:,}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Gradient accumulation: {CONFIG['gradient_accumulation']}")
print(f"Effective batch: {effective_batch}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"\n⏱️  TIME ESTIMATES:")
print(f"   Steps per epoch: {steps_per_epoch:,}")
print(f"   Total steps: {total_steps:,}")
print(f"   Estimated time: {estimated_hours:.1f} hours ({estimated_hours*60:.0f} minutes)")
print(f"   Expected completion: ~{int(estimated_hours)} hours {int((estimated_hours % 1) * 60)} minutes")
print(f"\n💡 You'll see a progress bar below with time remaining")
print(f"💡 Leave this running - go study other subjects! 📚")
print("=" * 60 + "\n")

start_time = time.time()

# TRAIN!
trainer.train()

# Calculate actual time
end_time = time.time()
training_duration = end_time - start_time
hours = int(training_duration // 3600)
minutes = int((training_duration % 3600) // 60)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"⏱️  Actual time: {hours}h {minutes}m")
print(f"📊 Avg per epoch: {training_duration/CONFIG['num_epochs']/60:.1f} minutes")
print("=" * 60)

# ============================================================================
# CELL 9: Evaluate Model (Run Ninth, ~2-3 minutes)
# ============================================================================

print("\n📊 Final Evaluation on Validation Set...")

eval_results = trainer.evaluate()

print("\n✅ Evaluation Results:")
print("-" * 60)
for key, value in sorted(eval_results.items()):
    print(f"  {key}: {value:.4f}")
print("-" * 60)

# ============================================================================
# CELL 10: Save & Test Model (Run Tenth, ~2-3 minutes)
# ============================================================================

print("\n💾 Saving final model...")

output_path = "./my_academic_summarizer_scientific"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# Save training info
training_info = {
    "model": CONFIG["model_name"],
    "datasets": {
        "scientific_papers": CONFIG["scientific_samples"],
        "booksum": CONFIG["booksum_samples"],
        "wikihow": CONFIG["wikihow_samples"],
        "total": CONFIG["total_samples"]
    },
    "final_metrics": {k: float(v) for k, v in eval_results.items()},
    "training_epochs": CONFIG["num_epochs"],
    "batch_size": CONFIG["batch_size"] * CONFIG["gradient_accumulation"],
    "training_time_minutes": training_duration / 60
}

with open(f"{output_path}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print(f"✅ Model saved to: {output_path}")

# Test the model
print("\n🧪 Testing on sample documents...\n")

test_examples = [
    {
        "type": "Course Policy",
        "text": """Cloud Computing (CC-702IT0C026) is a comprehensive course designed for B.Tech and MBA students 
in their seventh semester, specifically targeting programs in TECH IT, Computer Engineering, 
Artificial Intelligence & Data Science, Computer Science (Data Science), and Electronics & 
Telecommunication. The course is scheduled for the academic year 2025-26 and requires 
Computer Networks as a mandatory prerequisite, ensuring students have the foundational 
knowledge necessary to grasp advanced cloud computing concepts. The course covers various 
aspects including cloud service models (IaaS, PaaS, SaaS), deployment models (public, private, 
hybrid), virtualization technologies, and cloud security."""
    },
    {
        "type": "Research Paper",
        "text": """This paper presents a novel approach to distributed machine learning using federated learning 
techniques. The proposed methodology addresses privacy concerns in healthcare data analysis 
by enabling collaborative model training without centralizing sensitive patient information. 
Our experiments demonstrate a 23% improvement in model accuracy while maintaining strict 
privacy guarantees through differential privacy mechanisms."""
    }
]

for example in test_examples:
    print(f"{'='*60}")
    print(f"📄 {example['type']}")
    print(f"{'='*60}")
    
    inputs = tokenizer(
        "summarize scientific paper: " + example['text'],
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(model.device)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=256,
        num_beams=6,
        length_penalty=1.2,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"📝 Summary: {summary}\n")

# ============================================================================
# CELL 11: Create Download ZIP (Run Last, ~1 minute)
# ============================================================================

print("\n📦 Creating downloadable archive...")

!zip -r academic_summarizer_scientific.zip {output_path}

print("\n" + "=" * 60)
print("🎉 FINE-TUNING COMPLETE!")
print("=" * 60)
print(f"\n📦 Download: academic_summarizer_scientific.zip")
print(f"📁 Location: {output_path}")
print(f"\n📊 Performance:")
print(f"  ROUGE-L: {eval_results['eval_rougeL']:.4f}")
print(f"  ROUGE-1: {eval_results['eval_rouge1']:.4f}")
print(f"  ROUGE-2: {eval_results['eval_rouge2']:.4f}")
print(f"\n⏱️  Training Time: {hours}h {minutes}m")
print(f"\n🎯 Model trained on:")
print(f"  • 70% Scientific Papers (academic depth)")
print(f"  • 20% BookSum (long documents)")
print(f"  • 10% WikiHow (clear structure)")
print(f"\n📥 NEXT STEPS:")
print(f"  1. Download academic_summarizer_scientific.zip")
print(f"  2. Extract on your computer")
print(f"  3. Replace ./my_final_cnn_model folder")
print(f"  4. Restart server → Upload notes → Get PERFECT summaries!")
print("=" * 60)
