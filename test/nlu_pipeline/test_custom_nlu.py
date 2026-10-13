# test_intent.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

MODEL_DIR = "./models/vinai-phobert-base"
LABELS = ["dat_ve", "huy_ve", "sua_ve", "dong_y", "tu_choi", "xin_chao", "cung_cap"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def name(i):
    return LABELS[i]


def predict(texts, top_k=3):

    start_time = time.perf_counter()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    print(f"tokenizer in {(time.perf_counter() - start_time):.4f} seconds")

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(**batch).logits
    probs = torch.softmax(logits, dim=-1)
    print(f"feature extract in {(time.perf_counter() - start_time):.4f} seconds")

    results = []
    for text, p in zip(texts, probs):
        top = torch.topk(p, k=min(top_k, p.numel()))
        results.append({
            "text": text,
            "intent": name(top.indices[0].item()),
            "confidence": top.values[0].item(),
            "top_k": [(name(i.item()), round(v.item(), 3))
                    for v, i in zip(top.values, top.indices)],
        })
        
    return results


# (câu, intent mong đợi). None = câu mơ hồ, chỉ quan sát, không chấm.
TESTS = [
    ("Đặt hộ mình một chuyến xe 7 chỗ về Nam Định sáng mai nhé"),
    ("Hủy giúp tôi đơn đặt xe đi Nội Bài lúc 2h chiều nay với"),
    ("Cho anh dời lịch đón sang 5h chiều thay vì 4h được không"),
    ("Ok chốt mức giá này nha bạn, book xe giúp mình"),
    ("Giá này cao quá thôi mình chọn bên khác vậy, cảm ơn"),
    ("Dạ alo xin chào nhà xe ạ"),
    ("Đón tôi ở số 45 Nguyễn Trãi, Thanh Xuân nhé"),
    ("Cho mình book một xe chạy thẳng đi Hạ Long chiều nay"),
    ("Thôi khỏi cần đặt nữa đâu em nhé"),
    ("SĐT người đi là 0987654321 nha shop"),
    # Câu mơ hồ / đa ý: không chấm, chỉ xem model nghiêng về đâu
    ("Bên mình có xe đi Hải Phòng sáng mai không?"),
    ("Alo chào em, giá đi Nội Bài bao nhiêu vậy"),
    ("Thôi không hủy nữa đâu, giữ nguyên lịch cũ giúp mình"),
    ("Đổi sang 5h chiều được không"),
    ("Tôi nhắn số điện thoại 0912345678 rồi đó, chốt đặt xe nha"),
    ("Không, mình đổi ý rồi, dời sang ngày mai chứ hôm nay không đi nữa"),
    ("Dạ vâng chào bạn, hủy giúp mình chuyến xe vừa book nhé"),
    ("Xe 4 chỗ thì giá đó ok, chứ 7 chỗ thì đắt quá không đi"),
    ("Địa chỉ đón ở 12 Duy Tân, mà thôi để mình xem lại đã"),
    ("Cứ giữ xe đó cho anh, tí anh gửi địa chỉ sau"),
]

if __name__ == "__main__":
    results = predict([t for t in TESTS])
    correct = total = 0
    for r in results:
        # print(r)
        print(f"{r['text']}")
        print(f"-> intent: {r['intent']} ({r['confidence']:.3f})")
        print(f"-> top: {r['top_k']}")