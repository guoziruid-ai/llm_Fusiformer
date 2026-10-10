import json
from collections import Counter, defaultdict

DATA_PATH = "data/my_dataset.json"   # 改成你的数据集路径

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"样本总数: {len(data)}")

# =========================
# 1. 查看前几个晶体具体有哪些字段
# =========================
print("\n========== 前 5 个晶体字段 ==========")

for i, item in enumerate(data[:5]):
    print(f"\n样本 {i}")
    print("字段:", list(item.keys()))

    # 同时打印字段类型和部分内容
    for key, value in item.items():
        value_preview = str(value)
        if len(value_preview) > 150:
            value_preview = value_preview[:150] + "..."

        print(
            f"  {key:30s} "
            f"type={type(value).__name__:10s} "
            f"value={value_preview}"
        )


# =========================
# 2. 查看整个数据集出现过哪些字段
# =========================
all_fields = set()

for item in data:
    all_fields.update(item.keys())

print("\n========== 数据集全部字段 ==========")

for field in sorted(all_fields):
    print(field)


# =========================
# 3. 统计每个字段出现了多少次
# =========================
field_counts = Counter()

for item in data:
    field_counts.update(item.keys())

print("\n========== 字段覆盖情况 ==========")

for field in sorted(all_fields):
    count = field_counts[field]
    ratio = count / len(data) * 100

    print(
        f"{field:30s} "
        f"{count:8d}/{len(data):8d} "
        f"({ratio:6.2f}%)"
    )


# =========================
# 4. 查看不同样本是否具有不同字段组合
# =========================
schema_counter = Counter()

for item in data:
    schema = tuple(sorted(item.keys()))
    schema_counter[schema] += 1

print("\n========== 不同字段组合 ==========")
print(f"共有 {len(schema_counter)} 种字段组合")

for i, (schema, count) in enumerate(schema_counter.most_common(), 1):
    print(f"\n字段组合 {i}: {count} 个样本")
    print(schema)