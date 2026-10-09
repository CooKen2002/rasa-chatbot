import torch
from transformers import AutoTokenizer, AutoModel

M = "./models_hf/minilm-finetuned"   # hoặc model gốc để so sánh
# M = "./models_hf/minilm-merge"
# M = "./models_hf/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
tok, model = AutoTokenizer.from_pretrained(M), AutoModel.from_pretrained(M).eval()

# x = tok("Hủy vé đã đặt ở hà nội", return_tensors="pt")
# print(tok.convert_ids_to_tokens(x["input_ids"][0]))     # các token thật
words = ["hủy_vé", "đã", "đặt", "ở", "hà_nội"]
for_model = [w.replace("_", " ") for w in words]   # MiniLM không hiểu dấu "_"

enc = tok(for_model, is_split_into_words=True, return_tensors="pt")
print(tok.convert_ids_to_tokens(enc["input_ids"][0]))   # các mảnh thật
wid = enc.word_ids()
print(wid) 
with torch.no_grad():
    h = model(**enc).last_hidden_state
print(h.shape)                                           # (1, số_token, 384)
print(h[0, 2])                                       # 8 số đầu của vector token thứ 2

m = enc["attention_mask"].unsqueeze(-1).float()
sent = (h * m).sum(1) / m.sum(1)
print(sent.shape)      