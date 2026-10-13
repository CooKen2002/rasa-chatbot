import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

from .tokenizer import tokenizer

def parse(text):
        """
        Hàm xử lý chính của Pipeline NLU: Nhận đầu vào là text, trả về Token, Intent và Entities.
        """
        # Bước 1: Tokenization
        tokens = tokenizer(text)
        
        # Bước 2 & 3: Feature Extraction + Intent Classification
        # Pipeline của sklearn tự động đi qua TfidfVectorizer (Feature Extractor) rồi đến LogisticRegression
        predicted_intent = intent_pipeline.predict([text])[0]
        intent_probabilities = intent_pipeline.predict_proba([text])[0]
        confidence = float(np.max(intent_probabilities))
        
        # Bước 4: Entity Recognition
        entities = extract_entities(text)
        
        # Đóng gói kết quả đầu ra chuẩn dạng NLU JSON
        result = {
            "text": text,
            "tokens": tokens,
            "intent": {
                "name": predicted_intent,
                "confidence": round(confidence, 2)
            },
            "entities": entities
        }
        
        return result

if __name__ == "__main__":
    test_sentences = [
        "Cho mình đặt vé đi Hà Nội ngày 25/06 nhé",
        "Thời tiết hôm nay ở Đà Nẵng ra sao",
        "Chào bot, đặt hộ mình chuyến bay tới TP Hồ Chí Minh ngày mai"
    ]
    for sentence in test_sentences:
        output = parse(sentence)
        print(f"\nCâu gốc: {output['text']}")
        print(f"-> Tokens: {output['tokens']}")
        print(f"-> Intent: {output['intent']['name']} (Độ tự tin: {output['intent']['confidence']})")
        print(f"-> Entities: {output['entities']}")
    pass