import json

with open("file.json", "r") as f:
    data = json.load(f)

print(data[0])