"""LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the patient-grouped TRAIN split (completion-only loss).
Run:  python -m src.train --epochs 3
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from .data import get_split, chat_messages
from .model_utils import TARGET, load_tokenizer, ROOT

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--accum", type=int, default=8)
p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="checkpoints/lora_best")
p.add_argument("--log", default="results/train_log.json")
a = p.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

tok = load_tokenizer()
tr, va = get_split("train"), get_split("val")


def encode(row):
    prompt = tok.apply_chat_template(chat_messages(row), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(chat_messages(row, True), tokenize=False)
    pi = tok(prompt).input_ids
    fi = tok(full).input_ids
    assert fi[:len(pi)] == pi
    labels = [-100] * len(pi) + fi[len(pi):]
    return dict(ids=fi[:a.max_len], labels=labels[:a.max_len], truncated=len(fi) > a.max_len, n=len(fi))


enc_tr = [encode(r) for _, r in tr.iterrows()]
enc_va = [encode(r) for _, r in va.iterrows()]
lens = [e["n"] for e in enc_tr + enc_va]
trunc_info = dict(max_len=a.max_len, max_tokens_seen=max(lens), mean_tokens=float(np.mean(lens)),
                  n_truncated=int(sum(e["truncated"] for e in enc_tr + enc_va)))
print("token stats", trunc_info)

model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float16).cuda()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
for n_, p_ in model.named_parameters():
    if p_.requires_grad:
        p_.data = p_.data.float()  # fp32 adapter weights, fp16 frozen base
model.print_trainable_parameters()
params = [p_ for p_ in model.parameters() if p_.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
steps_total = a.epochs * math.ceil(len(enc_tr) / a.accum)
warm = max(1, int(0.1 * steps_total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
scaler = torch.amp.GradScaler("cuda")


def loss_of(e):
    ids = torch.tensor([e["ids"]]).cuda()
    lab = torch.tensor([e["labels"]]).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        return model(input_ids=ids, labels=lab).loss


@torch.no_grad()
def val_loss():
    model.eval()
    t = [loss_of(e).item() for e in enc_va]
    model.train()
    return float(np.mean(t))


log = dict(config=vars(a), truncation=trunc_info, steps=[], epochs=[])
best = 1e9
step = 0
model.train()
t0 = time.time()
for ep in range(a.epochs):
    order = np.random.permutation(len(enc_tr))
    run = []
    for i, idx in enumerate(order):
        l = loss_of(enc_tr[idx])
        scaler.scale(l / a.accum).backward()
        run.append(l.item())
        if (i + 1) % a.accum == 0 or i == len(order) - 1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step(); step += 1
            log["steps"].append(dict(step=step, train_loss=float(np.mean(run[-a.accum:])), lr=sched.get_last_lr()[0]))
            if step % 5 == 0:
                print(f"ep{ep} step{step} loss {log['steps'][-1]['train_loss']:.4f}", flush=True)
    vl = val_loss()
    log["epochs"].append(dict(epoch=ep + 1, train_loss=float(np.mean(run)), val_loss=vl, elapsed_s=time.time() - t0,
                              peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20))
    print(log["epochs"][-1], flush=True)
    if vl < best:
        best = vl
        model.save_pretrained(ROOT / a.out)
        log["best_epoch"] = ep + 1
    Path(ROOT / a.log).write_text(json.dumps(log, indent=1))
print("done, best val loss", best)
