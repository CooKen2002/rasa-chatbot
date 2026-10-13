import re

def extract_entities(text):
        """
        Thành phần 4: Entity Recognizer
        Sử dụng biểu thức chính quy (Regex) để trích xuất thực thể.
        """
        entities = []
        for entity_name, pattern in entity_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Lấy giá trị phần capture group bên trong pattern
                entity_value = match.group(1).strip()
                entities.append({
                    "entity": entity_name,
                    "value": entity_value,
                    "start": match.start(1),
                    "end": match.end(1)
                })
        return entities