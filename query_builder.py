"""
query_builder.py — Vietnamese Claim-Aware Query Builder
========================================================
Trích xuất thông tin quan trọng từ bài báo tiếng Việt để tạo
các truy vấn tìm kiếm có độ chính xác cao.

Chiến lược:
  1. Trích xuất Named Entities: người, tổ chức, địa danh (regex heuristic)
  2. Trích xuất số liệu & thống kê (%, tỷ đồng, triệu, tỷ lệ...)
  3. Trích xuất ngày tháng năm
  4. Trích xuất câu có claim cao (câu chứa động từ phát ngôn, số liệu)
  5. Tạo 2-3 queries có mục tiêu khác nhau → coverage rộng hơn
"""

import re
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
# Patterns tiếng Việt
# ─────────────────────────────────────────────────────────────

# Viết Hoa đầu từ = dấu hiệu proper noun (áp dụng sau tách câu)
_UPPER_WORD = re.compile(r"\b[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+")

# Số liệu & thống kê
_STATS = re.compile(
    r"""(?x)
    \d+(?:[.,]\d+)*\s*(?:%|phần\s*trăm|tỷ\s*đồng|triệu\s*đồng|tỷ\s*USD|
        nghìn\s*tỷ|triệu\s*người|nghìn\s*người|tấn|MW|km|ha|m²)
    |\b(?:tăng|giảm|đạt|chiếm|vượt|gần|hơn|khoảng|ước|dự\s*kiến)\s+\d+
    |\d+/\d+/\d{4}                   # date dd/mm/yyyy
    """,
    re.UNICODE
)

# Ngày tháng năm
_DATE = re.compile(
    r"""(?x)
    (?:ngày\s+)?\d{1,2}/\d{1,2}/\d{4}
    |tháng\s+\d{1,2}(?:\s+năm)?\s+\d{4}
    |năm\s+\d{4}
    |(?:thứ\s+(?:hai|ba|tư|năm|sáu|bảy|bảy|chủ\s+nhật))[,\s]+ngày\s+\d+
    """,
    re.UNICODE | re.IGNORECASE
)

# Động từ phát ngôn → câu có claim
_CLAIM_VERBS = re.compile(
    r"\b(?:cho\s+biết|khẳng\s+định|tuyên\s+bố|thông\s+báo|nói\s+rằng|"
    r"nhấn\s+mạnh|xác\s+nhận|phủ\s+nhận|cảnh\s+báo|kêu\s+gọi|"
    r"phát\s+hiện|công\s+bố|báo\s+cáo|ước\s+tính|nghiên\s+cứu)\b",
    re.UNICODE | re.IGNORECASE
)

# Tổ chức hay gặp trong tin Việt
_ORG_PATTERNS = re.compile(
    r"\b(?:Bộ\s+\w+|Chính\s+phủ|Quốc\s+hội|UBND|HĐND|"
    r"WHO|UN|EU|NATO|ASEAN|IMF|WB|ADB|"
    r"Bộ\s+Y\s+tế|Bộ\s+Công\s+an|Bộ\s+Giáo\s+dục|"
    r"Bệnh\s+viện\s+\w+|Trường\s+\w+|Đại\s+học\s+\w+|"
    r"Tập\s+đoàn\s+\w+|Công\s+ty\s+\w+)\b",
    re.UNICODE
)

# Stopwords để lọc
_STOPWORDS = {
    "và", "của", "là", "có", "trong", "để", "với", "được", "cho",
    "không", "này", "đã", "về", "các", "một", "những", "như", "đó",
    "theo", "khi", "từ", "tại", "vào", "ra", "lên", "xuống", "cũng",
    "thì", "mà", "hay", "hoặc", "nhưng", "vì", "nên", "còn", "đến",
    "bị", "do", "qua", "sau", "trước", "giữa", "trên", "dưới",
    "sẽ", "đang", "đây", "rằng", "rất", "nhiều", "ít", "lại",
    "ngay", "chỉ", "cả", "mọi", "gì", "ai", "nào", "bao", "thế",
    "tuy", "dù", "nếu", "kể", "hơn", "nữa", "được", "bởi", "vẫn",
}


def split_sentences(text: str) -> List[str]:
    """Tách văn bản thành câu dựa trên dấu câu kết thúc."""
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def extract_named_entities(text: str) -> List[str]:
    """
    Heuristic extraction của proper nouns tiếng Việt.
    Lấy chuỗi 1-3 từ hoa liên tiếp (NER đơn giản không cần model).
    """
    words = text.split()
    entities = []
    buf = []

    for w in words:
        clean = re.sub(r"[^\w]", "", w)
        if _UPPER_WORD.match(clean) and len(clean) > 1:
            buf.append(clean)
        else:
            if len(buf) >= 1:
                entity = " ".join(buf[:4])  # tối đa 4 từ
                if len(entity) > 3:
                    entities.append(entity)
            buf = []

    if buf:
        entities.append(" ".join(buf[:4]))

    # Thêm tổ chức từ pattern cụ thể
    for m in _ORG_PATTERNS.finditer(text):
        entities.append(m.group(0).strip())

    # Dedup giữ thứ tự
    seen = set()
    result = []
    for e in entities:
        key = e.lower().strip()
        if key not in seen and len(key) > 3:
            seen.add(key)
            result.append(e)

    return result[:10]  # Giới hạn 10 entity


def extract_statistics(text: str) -> List[str]:
    """Trích xuất cụm số liệu, tỷ lệ, thống kê."""
    results = []
    for m in _STATS.finditer(text):
        s = m.group(0).strip()
        if s:
            results.append(s)
    return results[:5]


def extract_dates(text: str) -> List[str]:
    """Trích xuất ngày tháng năm."""
    return [m.group(0).strip() for m in _DATE.finditer(text)][:3]


def score_sentence_claim(sentence: str) -> float:
    """
    Chấm điểm mức độ 'claim' của một câu.
    Câu có số liệu, động từ phát ngôn, entity → điểm cao.
    """
    score = 0.0
    if _CLAIM_VERBS.search(sentence):
        score += 2.0
    if _STATS.search(sentence):
        score += 1.5
    if _DATE.search(sentence):
        score += 0.5
    if _ORG_PATTERNS.search(sentence):
        score += 1.0
    # Độ dài hợp lý (không quá ngắn, không quá dài)
    n_words = len(sentence.split())
    if 8 <= n_words <= 40:
        score += 0.5
    return score


def get_key_claim_sentences(text: str, top_n: int = 3) -> List[str]:
    """Trả về top_n câu có claim cao nhất."""
    sentences = split_sentences(text)
    scored = [(s, score_sentence_claim(s)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n] if _ > 0]


def build_search_queries(text: str) -> List[Tuple[str, str]]:
    """
    Tạo danh sách (query_string, query_type) có mục tiêu khác nhau.

    Trả về list of (query, label):
      - ("Chính phủ GDP 2025 tăng trưởng", "entity_stats")
      - ("phiên họp thường kỳ tháng 5", "claim_sentence")
      - ("baochinhphu vnexpress GDP 2025", "broad")
    """
    queries = []

    entities = extract_named_entities(text)
    stats = extract_statistics(text)
    dates = extract_dates(text)
    claims = get_key_claim_sentences(text, top_n=2)

    # --- Query 1: Entity + Stats ---
    parts1 = []
    parts1.extend(entities[:3])
    parts1.extend(stats[:2])
    parts1.extend(dates[:1])
    q1 = " ".join(parts1).strip()
    if len(q1.split()) >= 2:
        queries.append((q1[:120], "entity_stats"))

    # --- Query 2: Claim sentence ngắn gọn ---
    if claims:
        # Lấy câu claim đầu tiên, cắt 15 từ đầu
        claim_words = claims[0].split()[:15]
        q2 = " ".join(claim_words)
        queries.append((q2, "claim_sentence"))

    # --- Query 3: Broad fallback (key nouns từ toàn văn) ---
    all_words = text.split()
    # Lấy từ hoa (thường là noun quan trọng), bỏ stopword
    key_nouns = []
    for w in all_words:
        clean = re.sub(r"[^\w]", "", w)
        if (
            _UPPER_WORD.match(clean)
            and len(clean) >= 3
            and clean.lower() not in _STOPWORDS
        ):
            key_nouns.append(clean)
    # Dedup
    key_nouns = list(dict.fromkeys(key_nouns))[:8]
    if key_nouns:
        q3 = " ".join(key_nouns[:6])
        if q3 not in [q for q, _ in queries]:
            queries.append((q3, "broad_nouns"))

    # Fallback nếu tất cả trống
    if not queries:
        fallback_tokens = [
            w for w in text.split()
            if len(w) >= 4 and w.lower() not in _STOPWORDS
        ][:10]
        queries.append((" ".join(fallback_tokens), "fallback"))

    return queries


def build_best_snippet_sentence_aware(
    doc_text: str,
    query_tokens: List[str],
    max_words: int = 80,
) -> str:
    """
    Trích xuất snippet tốt hơn: ưu tiên câu hoàn chỉnh, không cắt giữa câu.
    Thuật toán:
      1. Tách theo câu
      2. Score từng câu theo overlap với query_tokens
      3. Lấy câu tốt nhất, mở rộng thêm câu liền kề nếu còn quota từ
    """
    sentences = split_sentences(doc_text)
    if not sentences:
        # Fallback: sliding window cũ
        words = doc_text.split()
        if len(words) <= max_words:
            return doc_text
        return " ".join(words[:max_words])

    query_set = set(t.lower() for t in query_tokens)

    # Score từng câu
    scored = []
    for i, sent in enumerate(sentences):
        words_s = sent.lower().split()
        overlap = sum(1 for w in words_s if re.sub(r"[^\w]", "", w) in query_set)
        claim_sc = score_sentence_claim(sent)
        scored.append((i, overlap + claim_sc * 0.5))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_idx = scored[0][0]

    # Xây dựng snippet xung quanh câu tốt nhất
    result_sents = [sentences[best_idx]]
    word_count = len(sentences[best_idx].split())

    # Thêm câu liền kề (trước/sau) nếu còn quota
    for offset in [1, -1, 2, -2]:
        ni = best_idx + offset
        if 0 <= ni < len(sentences):
            extra_words = len(sentences[ni].split())
            if word_count + extra_words <= max_words:
                if offset > 0:
                    result_sents.append(sentences[ni])
                else:
                    result_sents.insert(0, sentences[ni])
                word_count += extra_words

    return " ".join(result_sents)
