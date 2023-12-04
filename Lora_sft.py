# Import necessary packages for the fine-tuning process
import os  # Operating system functionalities
import torch
from torch import nn
from datasets import load_dataset  # Loading datasets for training
from transformers import (
    AutoModelForCausalLM,  # AutoModel for language modeling tasks
    AutoTokenizer,  # AutoTokenizer for tokenization
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,  # Configuration for BitsAndBytes
    HfArgumentParser,  # Argument parser for Hugging Face models
    TrainingArguments,  # Training arguments for model training
    pipeline,  # Creating pipelines for model inference
    logging,  # Logging information during training
)

from plt import plot_loss
from network import LlamaWithClassifier
from peft import LoraConfig, PeftModel  # Packages for parameter-efficient fine-tuning (PEFT)
from trl import SFTTrainer
from param_dict import *
from network import ClassificationHead1
from network import ClassificationHead2
from network import ClassificationHead4
from train_fn import CustomTrainer
from custom_sft import NewSftTrainer

# Step 1 : Load dataset (you can process it here)
dataset = load_dataset("json", data_files=dataset_name, split="train")
#dataset = load_dataset("json", data_files="data/processed/pqal_pmt_process.json", split="train")

# Step 2 :Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Step 3 :Check GPU compatibility with bfloat16
# if compute_dtype == torch.float16 and use_4bit:
#     major, _ = torch.cuda.get_device_capability()
#     if major >= 8:
#         print("=" * 80)
#         print("Your GPU supports bfloat16: accelerate training with bf16=True")
#         print("=" * 80)

# Step 4 :Load base model
# model = LlamaWithClassifier.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map=device_map
# )

model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)

# model.classification_head = nn.Linear(max_seq_length*4096, 3, bias=False).to(model.device)
model.classification_head = ClassificationHead4(input_dim=max_seq_length*5120, num_classes=3).to(model.device)

# print(model)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.update({"output_hidden_states": True})

# Step 5 :Load LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# print(model)
# Step 6 :Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    remove_unused_columns=False,
    # load_best_model_at_end=True,
)


# def process(examples):
#     return examples

# print(model.classification_head)
def formatting(example):
    return example['format_qa']


# ["prompt_quetsion","prompt_answer","final_decision"]
trainer = NewSftTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    # data_collator=process,
    formatting_func=formatting,
    # clf_weight=clf_weight
)

# Step 9 :Train model
train_result = trainer.train()
# train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
trainer.save_model()

# if trainer.is_world_process_zero() and model_args.plot_loss:
plot_loss(output_dir, keys=["loss"])

# Step 10 :Save trained model
torch.save(model.classification_head, os.path.join(output_dir,'clf_head.pth'))
# trainer.model.save_pretrained(new_model)
