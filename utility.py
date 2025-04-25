import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "meta-llama/Llama-3.1-8B"

def load_drugbank_dataset():
    dataset = load_dataset("SkyHuReal/DrugBank-Alpaca")

    full_dataset = dataset["train"]

    train_valtest = full_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_valtest["train"]
    valtest_dataset = train_valtest["test"]

    val_test = valtest_dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    return train_dataset, val_dataset, test_dataset

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    return AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map={"": 0},)