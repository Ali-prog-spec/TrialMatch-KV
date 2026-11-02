import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from .data import chat_messages

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "checkpoints" / "lora_best"


def load_tokenizer(name=TARGET):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    return tok


def load_model(name=TARGET, adapter=None, merge=True, dtype=torch.float16, device="cuda"):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter))
        if merge:
            m = m.merge_and_unload()
    return m.eval()


def encode_prompt(tok, row, device="cuda"):
    text = tok.apply_chat_template(chat_messages(row), tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(device)
