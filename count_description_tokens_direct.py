import json
import statistics
from transformers import AutoTokenizer


# ============================================================
# 只需要修改这里
# ============================================================

# JSON 数据集路径
JSON_FILE = r"/root/llm_fusiformer/data/dataset_final_merged_with_robocrys.json"

# 训练 Fusiformer 时实际使用的 tokenizer / 模型路径
# 可以是本地路径，例如：
# TOKENIZER_PATH = r"/root/autodl-tmp/Qwen2.5-0.5B-Instruct"
#
# 也可以是 Hugging Face 名称，例如：
# TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_PATH = r"/root/models/Qwen2.5-1.5B-Instruct"

# JSON 中描述晶体文本的字段名
DESCRIPTION_FIELD = "description"

# 显示 token 最长的前多少条样本
TOP_K = 10


# ============================================================
# 以下部分通常不需要修改
# ============================================================


def load_descriptions(json_path, field="description"):
    """
    从 JSON 文件中读取 description。

    支持常见的三种数据格式：

    1. 列表：
       [
           {"composition": "...", "description": "..."},
           ...
       ]

    2. data 列表：
       {
           "data": [
               {"composition": "...", "description": "..."},
               ...
           ]
       }

    3. 字典：
       {
           "sample_1": {"description": "..."},
           "sample_2": {"description": "..."}
       }
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = []
    missing_count = 0

    # --------------------------------------------------------
    # 情况 1：最外层是 list
    # --------------------------------------------------------
    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            text = item.get(field)

            if isinstance(text, str) and text.strip():
                descriptions.append((i, text))
            else:
                missing_count += 1

    # --------------------------------------------------------
    # 情况 2 / 3：最外层是 dict
    # --------------------------------------------------------
    elif isinstance(data, dict):

        # {"data": [...]}
        if "data" in data and isinstance(data["data"], list):

            for i, item in enumerate(data["data"]):
                if not isinstance(item, dict):
                    continue

                text = item.get(field)

                if isinstance(text, str) and text.strip():
                    descriptions.append((i, text))
                else:
                    missing_count += 1

        # {"sample_id": {...}}
        else:
            for key, item in data.items():

                if not isinstance(item, dict):
                    continue

                text = item.get(field)

                if isinstance(text, str) and text.strip():
                    descriptions.append((key, text))
                else:
                    missing_count += 1

    else:
        raise ValueError("无法识别 JSON 数据结构。")

    return descriptions, missing_count


def percentile(values, p):
    """计算百分位数。"""
    values = sorted(values)

    if not values:
        return None

    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def main():

    print("=" * 70)
    print("Description Token 长度统计")
    print("=" * 70)

    print(f"\nJSON 文件：")
    print(JSON_FILE)

    print(f"\nTokenizer：")
    print(TOKENIZER_PATH)

    # --------------------------------------------------------
    # 加载 tokenizer
    # --------------------------------------------------------
    print("\n正在加载 tokenizer ...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True
    )

    print("Tokenizer 加载完成。")

    # --------------------------------------------------------
    # 读取 description
    # --------------------------------------------------------
    print("\n正在读取数据集 ...")

    descriptions, missing_count = load_descriptions(
        JSON_FILE,
        DESCRIPTION_FIELD
    )

    if not descriptions:
        print(f"\n错误：没有找到有效的 '{DESCRIPTION_FIELD}' 字段。")
        return

    print(f"有效 description 数量：{len(descriptions)}")
    print(f"缺失/空 description 数量：{missing_count}")

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------
    print("\n正在统计 token 长度 ...")

    results = []

    for sample_id, text in descriptions:

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        results.append(
            {
                "id": sample_id,
                "token_length": len(token_ids),
                "text": text
            }
        )

    lengths = [item["token_length"] for item in results]

    # --------------------------------------------------------
    # 基础统计
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Description Token 长度统计结果")
    print("=" * 70)

    print(f"样本数量        : {len(lengths)}")
    print(f"最短 token 长度 : {min(lengths)}")
    print(f"最长 token 长度 : {max(lengths)}")
    print(f"平均 token 长度 : {statistics.mean(lengths):.2f}")
    print(f"中位数          : {statistics.median(lengths):.2f}")

    print(f"P50             : {percentile(lengths, 50):.2f}")
    print(f"P75             : {percentile(lengths, 75):.2f}")
    print(f"P90             : {percentile(lengths, 90):.2f}")
    print(f"P95             : {percentile(lengths, 95):.2f}")
    print(f"P99             : {percentile(lengths, 99):.2f}")

    # --------------------------------------------------------
    # max_length 覆盖率
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("不同 max_length 的样本覆盖情况")
    print("=" * 70)

    limits = [
        64,
        128,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048
    ]

    for limit in limits:

        count = sum(length <= limit for length in lengths)
        ratio = count / len(lengths) * 100

        truncated_count = len(lengths) - count

        print(
            f"max_length={limit:4d} | "
            f"完整保留 {count:6d} ({ratio:6.2f}%) | "
            f"会被截断 {truncated_count:6d}"
        )

    # --------------------------------------------------------
    # 最长的 TOP_K 条
    # --------------------------------------------------------
    results.sort(
        key=lambda x: x["token_length"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print(f"Token 最长的前 {min(TOP_K, len(results))} 条 description")
    print("=" * 70)

    for rank, item in enumerate(results[:TOP_K], start=1):

        preview = (
            item["text"]
            .replace("\n", " ")
            .replace("\r", " ")
        )

        if len(preview) > 150:
            preview = preview[:150] + "..."

        print(
            f"\n[{rank}] "
            f"id = {item['id']} | "
            f"token_length = {item['token_length']}"
        )

        print(preview)

    print("\n" + "=" * 70)
    print("统计完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
