# AutoModel: Tải mô hình cơ sở (base model), đầu ra là các hidden states (đại diện vector), chưa có đầu định vị (head) cho tác vụ cụ thể.
# AutoModelForSequenceClassification: Dùng cho phân loại văn bản (nhận diện cảm xúc, phân loại chủ đề).
# AutoModelForTokenClassification: Dùng cho phân loại thực thể (NER), gắn nhãn từ loại (POS tagging).
# AutoModelForCausalLM: Dùng cho mô hình sinh chữ tự hồi quy (như GPT, Llama, Mistral) phục vụ chatbot, viết văn bản.
# AutoModelForSeq2SeqLM: Dùng cho mô hình dạng Sequence-to-Sequence (như T5, BART) phục vụ dịch thuật, tóm tắt văn bản.
# AutoModelForQuestionAnswering: Dùng cho các bài toán trả lời câu hỏi dựa trên đoạn văn (Extractive QA).
# AutoModelForMaskedLM: Dùng cho các mô hình như BERT, RoBERTa để dự đoán từ bị che (Masked Language Modeling).

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)

import torch
model_path = "./models/sks-intent-limo-booking"

model = AutoModel.from_pretrained(model_path)
model_seq_csf = AutoModelForSequenceClassification.from_pretrained(model_path)
model_tok_csf = AutoModelForTokenClassification.from_pretrained(model_path)

token = AutoTokenizer.from_pretrained(model_path)

text = "Tạm biệt"

inputs = token(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# print("Shape của last_hidden_state:", outputs.last_hidden_state.shape)

# from safetensors import safe_open
# with safe_open(model_path, "pt") as f:
#     print([k for k in f.keys() if "classifier" in k])

import os, time
from safetensors import safe_open

path = "./models/sks-intent-limo-booking/model.safetensors"
print("Sửa lần cuối:", time.ctime(os.path.getmtime(path)))

with safe_open(path, "pt") as f:
    w = f.get_tensor("classifier.weight")
    b = f.get_tensor("classifier.bias")
print("weight std:", w.std().item(), "| bias abs max:", b.abs().max().item())

from transformers import AutoTokenizer
print(token.tokenize("tôi muốn đặt xe đi sân bay"))
print(token.tokenize("hủy giúp mình chuyến xe đã đặt"))