import os
import traceback
from datetime import datetime

import transformers
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer

import wandb
from utility import load_drugbank_dataset, base_model_id, load_model

max_length = 400


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


IGNORE_INDEX = -100  # tokens with this label are ignored during loss calculation


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""Extract drug entities and inter-entity relationships from input and output them in the form of triplets <drug entity1, relationship, drug entity2>.

### input:
{data_point["input"]}

### output:
{data_point["output"]}
"""

    tokenized_full_prompt = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    tokenized_output = tokenizer(
        data_point["output"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    labels = [IGNORE_INDEX] * len(tokenized_full_prompt["input_ids"])

    # Find where the output starts and assign labels only for output tokens
    output_start = len(tokenized_full_prompt["input_ids"]) - len(tokenized_output["input_ids"])

    labels[output_start:] = tokenized_full_prompt["input_ids"][output_start:]

    tokenized_full_prompt["labels"] = labels

    return tokenized_full_prompt


try:
    train_dataset, val_dataset, test_dataset = load_drugbank_dataset()
    model = load_model()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.to("cuda:0")

    accelerator = Accelerator()
    model = accelerator.prepare_model(model)

    wandb.login()
    wandb_project = "drugbank-finetune"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    tokenized_train_dataset.set_format("torch")
    tokenized_val_dataset.set_format("torch")

    project = "drug-interaction"
    base_model_name = "llama-3.1-8B"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=2000,
            learning_rate=2.5e-5,
            logging_steps=50,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=25,
            eval_strategy="steps",
            eval_steps=50,
            do_eval=True,
            report_to="wandb",
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()

except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()
    exit(1)
