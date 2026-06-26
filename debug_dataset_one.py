from loader.loader import get_dataset
from loader.transforms import Compose, AddNodeFeature, AddEdgeFeature, AddAngleFeature, GraphCollecting

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

data_path = "ROOT/data/my_dataset.json"

transform = Compose([
    AddNodeFeature(),
    AddEdgeFeature(),
    AddAngleFeature(),
    GraphCollecting(["graph", "line_graph"], ["e_form"])
])

train_dataset, val_dataset, test_dataset = get_dataset(
    data_path=data_path,
    ratio_train_val_test=None,
    n_train_val_test=[10, 0, 0],
    transforms=transform,
    graph=True,
    line_graph=True,
)

local_idx = 2

print("\n========== Dataset basic info ==========")
print("train_dataset length:", len(train_dataset))
print("val_dataset:", val_dataset)
print("test_dataset:", test_dataset)
print("local_idx:", local_idx)

if local_idx < 0 or local_idx >= len(train_dataset):
    raise IndexError(
        f"local_idx={local_idx} 超出 train_dataset 范围。"
        f"当前 train_dataset 长度是 {len(train_dataset)}，"
        f"合法范围是 0 到 {len(train_dataset) - 1}。"
    )

# 关键：先直接查看原始样本，不触发 __getitem__
raw_info = train_dataset.data[local_idx]

print("\n========== Raw sample before __getitem__ ==========")
print("mp_id:", raw_info.get("mp_id"))
print("formula:", raw_info.get("formula"))

desc = raw_info.get("description", None)
print("description type:", type(desc))
print("description is None:", desc is None)
print("description repr:", repr(desc)[:300])

print("available keys:", raw_info.keys())

try:
    inputs, target = train_dataset[local_idx]

    print("\n========== After train_dataset[local_idx] ==========")
    print("inputs type:", type(inputs))
    print("inputs keys:", inputs.keys())
    print("target:", target)
    print("target shape:", target.shape)

    print("\n========== Graph input ==========")
    print("graph_input type:", type(inputs["graph_input"]))
    print("graph_input len:", len(inputs["graph_input"]))

    if "text" in inputs:
        print("\n========== Text input ==========")
        print("text exists: True")
        print("text keys:", inputs["text"].keys())
        print("input_ids shape:", inputs["text"]["input_ids"].shape)
        print("attention_mask shape:", inputs["text"]["attention_mask"].shape)
    else:
        print("\n========== Text input ==========")
        print("text exists: False")

except Exception as e:
    print("\n========== ERROR while calling train_dataset[local_idx] ==========")
    print("local index in train_dataset:", local_idx)
    print("mp_id:", raw_info.get("mp_id"))
    print("formula:", raw_info.get("formula"))
    print("description type:", type(desc))
    print("description repr:", repr(desc)[:500])
    print("error type:", type(e))
    print("error message:", e)
    raise