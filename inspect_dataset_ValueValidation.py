import json
import math
from collections import Counter

DATA_PATH = "data/my_dataset.json"

PROPERTY_FIELDS = [
    "e_form",
    "e_hull",
    "gap_pbe",
    "bulk_modulus",
    "shear_modulus",
    "mu_b",
]

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"总样本数: {len(data)}")

for field in PROPERTY_FIELDS:

    type_counter = Counter()
    valid = 0
    none_count = 0
    nan_count = 0
    invalid = 0

    values = []

    for item in data:

        value = item.get(field)

        type_counter[type(value).__name__] += 1

        if value is None:
            none_count += 1
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):

            if isinstance(value, float) and not math.isfinite(value):
                nan_count += 1
                continue

            valid += 1
            values.append(float(value))

        else:
            invalid += 1

    print("\n" + "=" * 70)
    print(field)
    print("=" * 70)

    print("数据类型:")
    for t, count in type_counter.items():
        print(f"  {t:15s}: {count}")

    print(f"\n有效数值 : {valid}")
    print(f"None     : {none_count}")
    print(f"NaN/Inf  : {nan_count}")
    print(f"其他无效 : {invalid}")

    if values:
        print(f"\nmin    = {min(values):.6g}")
        print(f"max    = {max(values):.6g}")
        print(f"mean   = {sum(values)/len(values):.6g}")