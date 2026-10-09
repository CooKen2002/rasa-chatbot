import torch
from transformers import AutoTokenizer, AutoModel

model_name = "./models_hf/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

text = "Đặt cho tôi một chuyến xe"
inputs = tokenizer(text, return_tensors="pt")


with torch.no_grad():
    outputs = model(**inputs)

# 1. Lấy ra tensor hidden states
# Kích thước: [Batch_size, Sequence_length, Hidden_dim] -> [1, 8, 768]
hidden_states = outputs.last_hidden_state

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print(f"Tổng số token: {len(tokens)}\n")

# 2. Duyệt qua từng token và xem vector 768 chiều của nó trông như thế nào
for i, token in enumerate(tokens):
    # Lấy vector 768 chiều của token thứ i
    token_vector = hidden_states[0, i]  # Kích thước: torch.Size([768])
    
    # In ra mặt chữ của token, 5 con số đầu tiên trong 768 số, và giá trị trung bình của vector
    print(f"Token [{token:^10}] -> 5 giá trị đầu: {token_vector[:5].tolist()}... (Độ dài vector: {len(token_vector)})")