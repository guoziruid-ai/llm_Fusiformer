import json
import math

data_path = "../llm_Fusiformer-master/data/my_dataset_wiki_prompt.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("top type:", type(data))

# 兼容 dict / list 两种 JSON 结构
if isinstance(data, dict):
    # 优先找常见容器字段
    for key in ["data", "records", "samples", "materials"]:
        if key in data and isinstance(data[key], list):
            records = data[key]
            print("using container key:", key)
            break
    else:
        # 如果 dict 本身是 {id: record}
        records = list(data.values())
        print("using dict values as records")
elif isinstance(data, list):
    records = data
    print("using list records")
else:
    raise TypeError(f"Unsupported json root type: {type(data)}")

print("num records:", len(records))

# 这里把你数据里可能的文本字段名都列出来
candidate_text_keys = [
    "text",
    "description",
    "desc",
    "summary",
    "formula",
    "composition",
    "material_text",
    "prompt",
]

bad = []

for i, rec in enumerate(records):
    if not isinstance(rec, dict):
        bad.append((i, "record_not_dict", type(rec).__name__, repr(rec)[:200]))
        continue

    found_key = None
    value = None

    for k in candidate_text_keys:
        if k in rec:
            found_key = k
            value = rec[k]
            break

    if found_key is None:
        bad.append((i, "missing_text_key", None, list(rec.keys())[:20]))
        continue

    ok = isinstance(value, str)

    # 空字符串也建议记录
    if ok and value.strip() == "":
        bad.append((i, f"empty_text:{found_key}", type(value).__name__, repr(value)))
        continue

    # NaN 特判
    if isinstance(value, float) and math.isnan(value):
        bad.append((i, f"nan_text:{found_key}", type(value).__name__, repr(value)))
        continue

    if not ok:
        bad.append((i, f"bad_type:{found_key}", type(value).__name__, repr(value)[:300]))

print("bad count:", len(bad))

for item in bad[:100]:
    print(item)