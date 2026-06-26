import torch
from transformers import AutoConfig

from loader.loader import get_dataset
from loader.transforms import (
    Compose,
    AddNodeFeature,
    AddEdgeFeature,
    AddAngleFeature,
    GraphCollecting,
)
from engine import to_device

try:
    from models.TextGuidedFusiformer import TextGuidedFusiformer
except ImportError:
    from TextGuidedFusiformer import TextGuidedFusiformer


# =======================
# 基本配置
# =======================
data_path = "./data/my_dataset.json"
target_name = "e_form"

# 先用 batch_size=1，避免 Qwen + DGL 同时上 GPU 时显存不够
batch_size = 1

qwen_model_name = "/root/models/Qwen2.5-1.5B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========== Device info ==========")
print("selected device:", device)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("current cuda device:", torch.cuda.current_device())
    print("cuda device name:", torch.cuda.get_device_name(0))


# =======================
# 构造 Dataset / DataLoader
# =======================
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

print("\n========== Before to_device ==========")
print("inputs keys:", inputs.keys())
print("targets shape:", targets.shape)
print("targets device:", targets.device)

assert "text" in inputs, "ERROR: inputs 中没有 text"
assert inputs["text"] is not None, "ERROR: inputs['text'] 是 None"

print("text keys:", inputs["text"].keys())
print("text input_ids shape:", inputs["text"]["input_ids"].shape)
print("text attention_mask shape:", inputs["text"]["attention_mask"].shape)
print("text attention_mask sum:", inputs["text"]["attention_mask"].sum(dim=1))
print("text input_ids device:", inputs["text"]["input_ids"].device)
print("text attention_mask device:", inputs["text"]["attention_mask"].device)


# =======================
# 移动到 GPU / CPU
# =======================
inputs = to_device(inputs, device)
targets = targets.to(device, non_blocking=True)

print("\n========== After to_device ==========")
print("targets device:", targets.device)
print("text input_ids device:", inputs["text"]["input_ids"].device)
print("text attention_mask device:", inputs["text"]["attention_mask"].device)
print("graph atomic_numbers device:", inputs["graph_input"][0].ndata["atomic_numbers"].device)
print("graph distance device:", inputs["graph_input"][0].edata["distance"].device)
print("line graph angle device:", inputs["graph_input"][1].edata["angle"].device)


# =======================
# 自动读取 Qwen hidden_size，避免 text_dim 写错
# =======================
print("\n========== Qwen config ==========")

config = AutoConfig.from_pretrained(qwen_model_name)
qwen_hidden_size = config.hidden_size

print("qwen model:", qwen_model_name)
print("qwen hidden_size:", qwen_hidden_size)

# 注意：
# Qwen2.5-1.5B hidden_size 一般是 1536。
# 如果模型里 text_dim 仍然默认 3584，会导致维度不匹配。
model = TextGuidedFusiformer(
    targets=[target_name],
    text_dim=qwen_hidden_size,
    qwen_model_name=qwen_model_name,
    freeze_qwen=True,
)

model = model.to(device)
model.eval()

print("\n========== Model info ==========")
print("model device example:", next(model.parameters()).device)
print("model.training:", model.training)

if hasattr(model, "qwen"):
    print("model.qwen exists: True")
    print("model.qwen.config.hidden_size:", model.qwen.config.hidden_size)
else:
    raise RuntimeError("ERROR: model 中没有 qwen 模块")

if hasattr(model, "text_proj"):
    print("model.text_proj:", model.text_proj)
else:
    raise RuntimeError("ERROR: model 中没有 text_proj 模块")


# =======================
# 替换 encode_text，加详细 debug 输出
# =======================
def debug_encode_text(text_tokens):
    print("\n========== ENTER model.encode_text() ==========")

    print("text_tokens type:", type(text_tokens))
    print("text_tokens keys:", text_tokens.keys())

    input_ids = text_tokens["input_ids"]
    attention_mask = text_tokens.get("attention_mask", None)

    print("input_ids shape:", input_ids.shape)
    print("input_ids device:", input_ids.device)

    if attention_mask is not None:
        print("attention_mask shape:", attention_mask.shape)
        print("attention_mask device:", attention_mask.device)
        print("attention_mask sum per sample:", attention_mask.sum(dim=1))
    else:
        print("attention_mask is None")

    with torch.no_grad():
        outputs = model.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.last_hidden_state

    print("\n========== Qwen output ==========")
    print("last_hidden_state shape:", hidden_states.shape)
    print("last_hidden_state device:", hidden_states.device)
    print("last_hidden_state dtype:", hidden_states.dtype)

    # 检查是否出现 nan / inf
    print("last_hidden_state has nan:", torch.isnan(hidden_states).any().item())
    print("last_hidden_state has inf:", torch.isinf(hidden_states).any().item())

    if attention_mask is not None:
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        denominator = mask_expanded.sum(dim=1).clamp(min=1e-9)

        text_emb_before_proj = (hidden_states * mask_expanded).sum(dim=1) / denominator
    else:
        text_emb_before_proj = hidden_states.mean(dim=1)

    print("\n========== Text pooling ==========")
    print("text_emb before text_proj shape:", text_emb_before_proj.shape)
    print("text_emb before text_proj device:", text_emb_before_proj.device)
    print("text_emb before text_proj dtype:", text_emb_before_proj.dtype)

    text_emb_after_proj = model.text_proj(text_emb_before_proj)

    print("\n========== Text projection ==========")
    print("text_emb after text_proj shape:", text_emb_after_proj.shape)
    print("text_emb after text_proj device:", text_emb_after_proj.device)
    print("text_emb after text_proj dtype:", text_emb_after_proj.dtype)
    print("text_emb after text_proj has nan:", torch.isnan(text_emb_after_proj).any().item())
    print("text_emb after text_proj has inf:", torch.isinf(text_emb_after_proj).any().item())

    print("========== EXIT model.encode_text() ==========\n")

    return text_emb_after_proj


# 用 debug 版本替换原来的 encode_text
model.encode_text = debug_encode_text


# =======================
# 运行完整 model forward
# =======================
print("\n========== Run full TextGuidedFusiformer.forward() ==========")

with torch.no_grad():
    outputs = model(inputs)

print("\n========== Model forward output ==========")
print("outputs shape:", outputs.shape)
print("outputs device:", outputs.device)
print("outputs dtype:", outputs.dtype)
print("outputs:", outputs)

print("targets shape:", targets.shape)
print("targets:", targets)

assert outputs.shape[0] == targets.shape[0], "ERROR: outputs 和 targets 的 batch size 不一致"

print("\nPASS: TextGuidedFusiformer.forward() 已经收到 text，Qwen 已经输出 text embedding，并完成完整 forward")

