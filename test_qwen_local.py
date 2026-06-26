import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModel

model_path = "/root/models/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    output_hidden_states=True,
    local_files_only=True
)

print("Loaded successfully.")
print("hidden size:", model.config.hidden_size)