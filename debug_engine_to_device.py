import torch

from loader.loader import get_dataset
from loader.transforms import (
    Compose,
    AddNodeFeature,
    AddEdgeFeature,
    AddAngleFeature,
    GraphCollecting,
)

from engine import to_device

data_path = "./data/my_dataset.json"
target_name = "e_form"

batch_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========== Device info ==========")
print("torch.cuda.is_available():", torch.cuda.is_available())
print("selected device:", device)

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

loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=train_dataset.collect(),
)

inputs, targets = next(iter(loader))

print("\n========== Before engine.to_device ==========")
print("inputs keys:", inputs.keys())
print("targets device:", targets.device)

print("graph atomic_numbers device:", inputs["graph_input"][0].ndata["atomic_numbers"].device)
print("graph distance device:", inputs["graph_input"][0].edata["distance"].device)
print("line graph angle device:", inputs["graph_input"][1].edata["angle"].device)

if "text" in inputs and inputs["text"] is not None:
    print("text input_ids device:", inputs["text"]["input_ids"].device)
    print("text attention_mask device:", inputs["text"]["attention_mask"].device)
    print("text input_ids shape:", inputs["text"]["input_ids"].shape)
    print("text attention_mask shape:", inputs["text"]["attention_mask"].shape)
else:
    raise RuntimeError("Before to_device: inputs['text'] 不存在或为 None")

inputs = to_device(inputs, device)
targets = targets.to(device, non_blocking=True)

print("\n========== After engine.to_device ==========")
print("targets device:", targets.device)

print("graph atomic_numbers device:", inputs["graph_input"][0].ndata["atomic_numbers"].device)
print("graph distance device:", inputs["graph_input"][0].edata["distance"].device)
print("line graph angle device:", inputs["graph_input"][1].edata["angle"].device)

print("text input_ids device:", inputs["text"]["input_ids"].device)
print("text attention_mask device:", inputs["text"]["attention_mask"].device)

print("text input_ids shape:", inputs["text"]["input_ids"].shape)
print("text attention_mask shape:", inputs["text"]["attention_mask"].shape)

print("\n========== Check result ==========")

if torch.cuda.is_available():
    assert inputs["text"]["input_ids"].device.type == "cuda", "input_ids 没有移动到 GPU"
    assert inputs["text"]["attention_mask"].device.type == "cuda", "attention_mask 没有移动到 GPU"
    assert targets.device.type == "cuda", "targets 没有移动到 GPU"

    assert inputs["graph_input"][0].ndata["atomic_numbers"].device.type == "cuda", "graph 节点特征没有移动到 GPU"
    assert inputs["graph_input"][0].edata["distance"].device.type == "cuda", "graph 边特征没有移动到 GPU"
    assert inputs["graph_input"][1].edata["angle"].device.type == "cuda", "line graph 角度特征没有移动到 GPU"

    print("PASS: engine.to_device() 已经把 graph、line_graph、text、targets 都移动到 GPU")
else:
    assert inputs["text"]["input_ids"].device.type == "cpu"
    assert inputs["text"]["attention_mask"].device.type == "cpu"
    print("PASS: 当前无 CUDA，数据保持在 CPU")