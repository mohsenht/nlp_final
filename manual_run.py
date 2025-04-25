import torch
from peft import PeftModel
from transformers import AutoTokenizer

from utility import base_model_id, load_model

base_model = load_model()

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

ft_model = PeftModel.from_pretrained(base_model, "llama-3.1-8B-drug-interaction/checkpoint-2000")
ft_model.to("cuda")

eval_prompt = """Extract drug entities and inter-entity relationships from input and output them in the form of triplets <drug entity1, relationship, drug entity2>.

### input:
The effects of nonbenzodiazepine agonists at benzodiazepine receptors, such as zopiclone, triazolopyridazines and others, are also blocked by ROMAZICON.

### output:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    generated_ids = ft_model.generate(
        **model_input,
        max_new_tokens=100
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
