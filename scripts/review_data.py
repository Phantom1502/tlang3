repo = "sullivan1502/zone-pretrain-ids-data"
from datasets import load_dataset

ds = load_dataset(repo, split="train")
print(ds)

# print a few samples
for i in range(5):
    print(ds[i]["input_ids"])
    print(ds[i]["labels"])