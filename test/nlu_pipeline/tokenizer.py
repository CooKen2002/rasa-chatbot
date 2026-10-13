import re
def tokenizer(self, text):
        # Chuyển về chữ thường và tách từ bằng khoảng trắng hoặc dấu câu cơ bản
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens