"""
Utility functions for MRCD Framework.
Text preprocessing, cleaning, seeding, and debug logging.
"""

import os
import re
import csv
import random
import unicodedata

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Thiết lập seed ngẫu nhiên để đảm bảo tính tái lập (reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt dùng chung cho cả SLM và MRCD inference.
    
    Các bước:
    1. Chuyển về chuỗi, chuẩn hóa Unicode (NFC).
    2. Chuyển thành chữ thường (lowercase).
    3. Loại bỏ URL, @mentions.
    4. Giữ lại chữ cái (kể cả có dấu), số, khoảng trắng, dấu #.
    5. Chuẩn hóa khoảng trắng.
    """
    text = str(text)
    # Chuẩn hóa Unicode: đưa các tổ hợp dấu về dạng ký tự đơn (viết liền)
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    # Loại bỏ URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Loại bỏ mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Giữ lại: chữ cái (có dấu), số, khoảng trắng, dấu #
    # re.UNICODE để \w bao gồm chữ cái tiếng Việt
    text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
    # Chuẩn hóa khoảng trắng (xóa space thừa)
    text = " ".join(text.split())
    return text


def clean_query(text: str) -> str:
    """
    Làm sạch truy vấn để gửi lên search engine:
    - Chuẩn hóa Unicode (NFKC)
    - Chỉ giữ chữ cái (có dấu), số, khoảng trắng
    - Xóa dấu câu, ký tự đặc biệt
    """
    text = unicodedata.normalize("NFKC", str(text))
    # Giữ lại chữ cái (kể cả có dấu), số, khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Cắt ngắn văn bản đến độ dài max_length, ưu tiên cắt tại ranh giới từ.
    """
    if len(text) <= max_length:
        return text
    cut_pos = text.rfind(" ", 0, max_length)
    if cut_pos == -1:
        return text[:max_length] + "..."
    return text[:cut_pos] + "..."


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng tổ hợp (NFC)."""
    return unicodedata.normalize('NFC', text)

# ---------- Regex patterns ----------
URL_PATTERN = r"http\S+|www\S+"
TAG_PATTERN = r"\[.*?\]"          # [hook], [quote], [img]...
HTML_PATTERN = r"<.*?>"
MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#(\w+)"

# Tối ưu lại EMOJI_PATTERN để bao quát hơn (tránh sót các icon rác)
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map symbols
    r"\U0001F1E0-\U0001F1FF"  # flags
    r"\U000E0000-\U000E007F"  # tags
    r"\u2600-\u27BF"          # Miscellaneous Symbols / Dingbats (❤️, ✨, ✔️...)
    r"]+",
    flags=re.UNICODE
)

REPEAT_PATTERN = r'(.)\1{2,}' 
def reduce_repeated_chars(text: str) -> str:
    """Giảm lặp ký tự (giữ tối đa 2 lần)."""
    return re.sub(REPEAT_PATTERN, r'\1\1', text)

def clean_text_transformer(text: str) -> str:
    """Tiền xử lý tổng hợp cho các mô hình (PhoBERT). Không xóa dấu câu hợp lệ."""
    if not isinstance(text, str):
        return ""

    # 1. Chuẩn hóa Unicode
    text = normalize_unicode(text)

    # 2. XÓA DẤU CÂU NỘI TỪ (Fix vụ "co.n m.ẹ", "b-ố l-á-o")
    # Chỉ xóa dấu câu (. , - * _ / :) nếu nó bị kẹp giữa 2 chữ cái tiếng Việt/English
    text = re.sub(r'(?<=[a-zA-Zà-ỹÀ-Ỹ])([.,\-*_\/:])(?=[a-zA-Zà-ỹÀ-Ỹ])', '', text)


    # 3. Loại bỏ mention
    text = re.sub(MENTION_PATTERN, "", text)

    # 4. Xử lý hashtag: giữ phần từ, bỏ #
    text = re.sub(HASHTAG_PATTERN, r'\1', text)

    # 5. Loại bỏ URL, tag, HTML
    text = re.sub(URL_PATTERN, "", text)
    text = re.sub(TAG_PATTERN, "", text)
    text = re.sub(HTML_PATTERN, "", text)

    # 6. Loại bỏ emoji (Đã nâng cấp pattern để sạch hơn)
    text = EMOJI_PATTERN.sub("", text)

    # [BỔ SUNG VÀO BƯỚC 7] XỬ LÝ DẤU CÂU HỢP LỆ (Thay vì xóa sạch)
    # 7.1. Gom các dấu câu lặp lại (Biến "!!!!" thành "!", "???" thành "?")
    text = re.sub(r'([.!?,])\1+', r'\1', text)
    
    # 7.2. Tách dấu câu dính liền chữ (Biến "được.Xong" thành "được. Xong", "học,chơi" thành "học, chơi")
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)

    # 8. Giảm lặp kí tự (Ví dụ: "ngonnnn" -> "ngon")
    text = reduce_repeated_chars(text)

    # 9. Chuẩn hóa khoảng trắng phát sinh do các bước xóa ở trên
    text = re.sub(r"\s+", " ", text).strip()

    return text





def clean_text_for_slm(text: str) -> str:
    """
    Tiền xử lý văn bản đầy đủ cho SLM (PhoBERT): clean_text_transformer + tách từ underthesea.
    
    Chỉ dùng hàm này trước khi đưa vào SLM inference/fine-tune.
    KHÔNG dùng cho nội dung RAG/wiki (tránh thêm dấu gạch dưới vào chunk_text).
    """
    text = clean_text_transformer(text)
    if not text:
        return text
    # Tách từ tiếng Việt bằng underthesea (tạo dạng "xung_đột" chuẩn cho PhoBERT)
    try:
        from underthesea import word_tokenize
        text = word_tokenize(text, format="text")
    except Exception:
        # Trong trường hợp chưa cài underthesea, giữ nguyên văn bản
        pass
    return text



def log_retrieval_to_csv(
    func_name: str,
    query: str,
    title: str,
    url: str,
    snippet: str,
    filepath: str = None,
):
    """Ghi nhật ký kết quả truy xuất vào file CSV để debug."""
    from src.config import RETRIEVAL_DEBUG_CSV

    filepath = filepath or RETRIEVAL_DEBUG_CSV
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["source_function", "query", "title", "url", "snippet"]
                )
            writer.writerow([func_name, query, title, url, snippet])
    except Exception:
        pass


def log_prediction_to_csv(
    event_id: int,
    text: str,
    label: int,
    conf: float,
    round_id: int,
    status: str,
    filepath: str = None,
):
    """Ghi kết quả dự đoán cuối cùng của một sự kiện vào file CSV."""
    from src.config import RESULTS_CSV

    filepath = filepath or RESULTS_CSV
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["event_id", "label", "confidence", "round", "status", "text_snippet"]
                )
            text_snippet = text.replace("\n", " ")
            writer.writerow([event_id, label, conf, round_id, status, text_snippet])
    except Exception:
        pass


def log_round_trace_to_csv(
    round_id: str | int,
    event_id: int,
    text: str,
    y_slm: int,
    y_llm: int,
    ground_truth: int | str,
    conf_slm: float,
    prompt: str,
    filepath: str = None,
):
    """Ghi nhật ký chi tiết từng vòng (trace) cho từng sự kiện."""
    from src.config import TRACE_CSV

    filepath = filepath or TRACE_CSV
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "round",
                        "event_id",
                        "y_slm",
                        "y_llm",
                        "ground_truth",
                        "conf_slm",
                        "input",
                        "prompt",
                    ]
                )
            writer.writerow(
                [
                    round_id,
                    event_id,
                    y_slm,
                    y_llm,
                    ground_truth if ground_truth is not None else "N/A",
                    f"{conf_slm:.4f}",
                    text,
                    str(prompt),
                ]
            )
    except Exception:
        pass