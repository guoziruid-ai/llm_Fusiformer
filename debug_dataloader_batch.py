import torch

from loader.loader import get_dataset
from loader.transforms import (
    Compose,
    AddNodeFeature,
    AddEdgeFeature,
    AddAngleFeature,
    GraphCollecting,
)

data_path = "./data/my_dataset.json"
target_name = "e_form"

batch_size = 2

transform = Compose([
    AddNodeFeature(),
    AddEdgeFeature(),
    AddAngleFeature(),
    GraphCollecting(["graph", "line_graph"], [target_name])
])

train_dataset, _, _ = get_dataset(
    data_path=data_path,
    ratio_train_val_test=None,
    n_train_val_test=[batch_size, 0, 0],
    transforms=transform,
    graph=True,
    line_graph=True,
)

print("\n========== Dataset info ==========")
print("train_dataset length:", len(train_dataset))

loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=train_dataset.collect(),
)

inputs, targets = next(iter(loader))

print("\n========== DataLoader batch output ==========")
print("inputs type:", type(inputs))
print("inputs keys:", inputs.keys())
print("targets shape:", targets.shape)
print("targets:", targets)

print("\n========== Graph input ==========")
print("graph_input exists:", "graph_input" in inputs)
print("graph_input type:", type(inputs["graph_input"]))
print("graph_input len:", len(inputs["graph_input"]))

batched_graph = inputs["graph_input"][0]
batched_line_graph = inputs["graph_input"][1]

print("batched graph:", batched_graph)
print("batched line graph:", batched_line_graph)

print("batched graph num nodes:", batched_graph.num_nodes())
print("batched graph num edges:", batched_graph.num_edges())
print("batched line graph num nodes:", batched_line_graph.num_nodes())
print("batched line graph num edges:", batched_line_graph.num_edges())

print("\n========== Text input ==========")

if "text" not in inputs:
    print("ERROR: inputs 中没有 text 字段")
elif inputs["text"] is None:
    print("ERROR: inputs['text'] is None")
else:
    text = inputs["text"]

    print("text exists: True")
    print("text type:", type(text))
    print("text keys:", text.keys())

    input_ids = text["input_ids"]
    attention_mask = text["attention_mask"]

    print("input_ids shape:", input_ids.shape)
    print("attention_mask shape:", attention_mask.shape)

    print("input_ids device:", input_ids.device)
    print("attention_mask device:", attention_mask.device)

    print("attention_mask sum per sample:", attention_mask.sum(dim=1))
    print("valid token ratio per sample:", attention_mask.sum(dim=1) / attention_mask.shape[1])

    print("first sample first 30 input_ids:")
    print(input_ids[0, :30])

    print("first sample first 80 attention_mask:")
    print(attention_mask[0, :80])

    print("first sample last 80 attention_mask:")
    print(attention_mask[0, -80:])

print("\n========== Check result ==========")

assert "text" in inputs, "DataLoader batch 后 inputs 中没有 text"
assert inputs["text"] is not None, "DataLoader batch 后 inputs['text'] 是 None"
assert "input_ids" in inputs["text"], "inputs['text'] 中没有 input_ids"
assert "attention_mask" in inputs["text"], "inputs['text'] 中没有 attention_mask"
assert inputs["text"]["input_ids"].shape[0] == batch_size, "batch 维度不等于 batch_size"
assert inputs["text"]["attention_mask"].shape[0] == batch_size, "batch 维度不等于 batch_size"

print("PASS: DataLoader.collect() 已经正确 batch 文本")