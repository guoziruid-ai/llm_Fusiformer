import json

with open("/data/home/gcc/Fusiformer/llm_Fusiformer/data/my_dataset.json", "r") as f:
    data = json.load(f)

print(len(data))