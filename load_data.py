from datasets import load_dataset

dataset = load_dataset("zwhe99/commonsense_170k")
dataset.save_to_disk("/root/autodl-tmp/data/commonsense_170k")
