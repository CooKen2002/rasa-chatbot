import re, random, yaml, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# MODEL = "./models_hf/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL = "./models_hf/minilm-merge"
NLU = "./test/data/nlu.yml"
OUT = "models_hf/minilm-finetuned"
random.seed(42); torch.manual_seed(42)

# ---------- 0. Đọc dữ liệu từ nlu.yml ----------
def load_nlu(path):
    data = {}
    for item in yaml.safe_load(open(path, encoding="utf-8")).get("nlu", []):
        if "intent" not in item:
            continue
        for line in item["examples"].splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)  # bỏ nhãn entity
                data.setdefault(item["intent"], []).append(line)
    return data

data = load_nlu(NLU)
train, val = [], []
for intent, xs in data.items():
    random.shuffle(xs)
    k = max(1, int(len(xs) * 0.2))
    val += [(x, intent) for x in xs[:k]]
    train += [(x, intent) for x in xs[k:]]
intents = sorted(data)
i2id = {l: i for i, l in enumerate(intents)}
print(f"{len(intents)} intent | train={len(train)} | val={len(val)}")

tok = AutoTokenizer.from_pretrained(MODEL)

# ---------- 1. Vector câu (mean pooling) ----------
@torch.no_grad()
def embed(encoder, texts, bs=32):
    encoder.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=64)
        h = encoder(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        out.append(F.normalize(v, dim=-1))
    return torch.cat(out)

# ---------- 2. Đoán intent từ weight gốc (chưa train gì) ----------
def centroid_eval(encoder, tag, show=()):
    tx = embed(encoder, [x for x, _ in train])
    ty = torch.tensor([i2id[l] for _, l in train])
    cents = F.normalize(torch.stack([tx[ty == i].mean(0) for i in range(len(intents))]), dim=-1)

    def predict(texts):
        sims = embed(encoder, texts) @ cents.T           # độ giống cosine với từng intent
        return F.softmax(sims * 20, dim=-1)              # 20 = nhiệt độ, chỉ để dễ đọc

    p = predict([x for x, _ in val])
    acc = (p.argmax(-1) == torch.tensor([i2id[l] for _, l in val])).float().mean().item()
    print(f"[{tag}] acc trên tập val = {acc:.3f}")
    for t in show:
        pr = predict([t])[0]
        top = pr.topk(min(3, len(intents)))
        print(f"   '{t}' ->", [(intents[i], round(s.item(), 3)) for s, i in zip(top.values, top.indices)])
    return acc

samples = [
            "Đặt cho cậu một chiếc xe về hai bà trưng",
            "Alo cho anh một chiếc xe di chuyển ra bến xe khách nhé",
            "Hủy vé đã đặt ở hà nội"
        ]

base = AutoModel.from_pretrained(MODEL)
print("Số tham số:", sum(p.numel() for p in base.parameters()) / 1e6, "triệu")
acc_before = centroid_eval(base, f"{MODEL}", samples)

# # ---------- 3. Fine-tune ----------
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL, num_labels=len(intents),
#     id2label=dict(enumerate(intents)), label2id=i2id)
# opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
# EPOCHS, BS = 5, 16

# def batches(pairs, shuffle):
#     pairs = pairs[:]
#     if shuffle: random.shuffle(pairs)
#     for i in range(0, len(pairs), BS):
#         chunk = pairs[i:i + BS]
#         b = tok([x for x, _ in chunk], return_tensors="pt", padding=True,
#                 truncation=True, max_length=64)
#         yield b, torch.tensor([i2id[l] for _, l in chunk])

# for ep in range(EPOCHS):
#     model.train(); total = 0
#     for b, y in batches(train, True):
#         loss = model(**b, labels=y).loss
#         loss.backward(); opt.step(); opt.zero_grad()
#         total += loss.item()
#     model.eval(); correct = 0
#     with torch.no_grad():
#         for b, y in batches(val, False):
#             correct += (model(**b).logits.argmax(-1) == y).sum().item()
#     print(f"epoch {ep + 1}: loss={total:.3f} | val acc={correct / len(val):.3f}")

# # ---------- 4. Đo lại bằng cùng phương pháp như bước 2, rồi lưu ----------
# acc_after = centroid_eval(model.base_model, "SAU fine-tune", samples)
# print(f"Cải thiện: {acc_before:.3f} -> {acc_after:.3f}")

# model.base_model.save_pretrained(OUT)   # chỉ lưu encoder, bỏ đầu phân loại
# tok.save_pretrained(OUT)
# print("Đã lưu vào", OUT)