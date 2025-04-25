import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from utility import load_drugbank_dataset, load_model, base_model_id


def tokenize(prompt):
    result = tokenizer(prompt)
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""Extract drug entities and inter-entity relationships from input and output them in the form of triplets <drug entity1, relationship, drug entity2>.

    ### input:
    {data_point["input"]}

    ### output:
    {data_point["output"]}
    """
    return tokenize(full_prompt)

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input')
    plt.show()

train_dataset, val_dataset, test_dataset = load_drugbank_dataset()
model = load_model()
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True,
)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt)

plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)