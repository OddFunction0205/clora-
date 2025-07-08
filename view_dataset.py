from datasets import load_dataset

dataset = load_dataset("zwhe99/commonsense_170k")
print(dataset["train"][0])
