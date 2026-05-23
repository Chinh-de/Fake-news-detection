"""
explainer.py — Explanation Generator for PhoBERT Fake News Detection
=====================================================================
Tạo giải thích dạng ngôn ngữ tự nhiên (tiếng Việt) giải thích tại sao
một tin được PhoBERT phân loại là THẬT hoặc GIẢ.

Luồng:
  1. Nhận kết quả predict (label, confidence, probabilities)
  2. Nhận evidence list từ retriever
  3. Kết hợp thành explanation có cấu trúc:
     - Kết luận chính (PhoBERT verdict)
     - Phân tích bằng chứng (Evidence distribution)
     - Các yếu tố quyết định (Key factors)
     - Cảnh báo / disclaimer
"""

from typing import List, Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────
# Ngưỡng độ tin cậy
# ─────────────────────────────────────────────────────────────
HIGH_CONF    = 0.85
MEDIUM_CONF  = 0.65
LOW_CONF     = 0.50


def _confidence_level(conf: float) -> str:
    if conf >= HIGH_CONF:
        return "rất cao"
    elif conf >= MEDIUM_CONF:
        return "khá cao"
    elif conf >= LOW_CONF:
        return "trung bình"
    else:
        return "thấp"


def _model_verdict_text(label: str, conf: float) -> str:
    """Câu phán quyết chính của mô hình PhoBERT."""
    level = _confidence_level(conf)
    if label == "FAKE":
        return (
            f"🔴 PhoBERT phân loại tin này là **TIN GIẢ** với độ tin cậy **{conf:.1%}** ({level}).\n\n"
            f"Mô hình phát hiện các dấu hiệu đặc trưng của nội dung thông tin sai lệch trong văn bản."
        )
    else:
        return (
            f"🟢 PhoBERT phân loại tin này là **TIN THẬT** với độ tin cậy **{conf:.1%}** ({level}).\n\n"
            f"Mô hình đánh giá văn bản có các đặc điểm phù hợp với tin tức chính thống, đáng tin cậy."
        )


def _evidence_analysis_text(distribution: Dict[str, Any], phobert_label: str) -> Tuple[str, bool]:
    """
    Phân tích phân phối nhãn của evidence.
    Returns: (text, is_consistent_with_model)
    """
    real_c    = distribution.get("real_count", 0)
    fake_c    = distribution.get("fake_count", 0)
    labeled_c = distribution.get("labeled_count", real_c + fake_c)
    web_c     = distribution.get("web_count", 0)
    bkai_c    = distribution.get("bkai_count", 0)
    total     = distribution.get("total", 0)
    dom       = distribution.get("dominant_label", "UNKNOWN")
    ratio     = distribution.get("agreement_ratio", 0.0)

    if total == 0:
        return "⚠️ Không tìm thấy bài báo nào liên quan.", False

    lines = []

    # Thống kê tổng quan
    parts = []
    if bkai_c:
        parts.append(f"**{bkai_c}** bài từ corpus BKAI")
    if web_c:
        parts.append(f"**{web_c}** bài từ Web RAG")
    lines.append(f"📊 Tìm được {' + '.join(parts)} ({total} tổng cộng).")

    # Phân phối Real/Fake (chỉ có nếu corpus có nhãn — ViFN fallback)
    is_consistent = False
    if labeled_c > 0:
        lines.append(f"")
        lines.append(f"Trong **{labeled_c}** bài có nhãn:")
        lines.append(f"- 🟢 **{real_c}** bài **TIN THẬT**")
        lines.append(f"- 🔴 **{fake_c}** bài **TIN GIẢ**")

        is_consistent = (dom == phobert_label)
        if is_consistent:
            lines.append(
                f"\n✅ **Bằng chứng nhất quán**: {ratio:.0%} bài có nhãn **{dom}** — "
                f"**phù hợp** với phán quyết của PhoBERT."
            )
        else:
            lines.append(
                f"\n⚡ **Bằng chứng mâu thuẫn**: đa số bài có nhãn **{dom}** — "
                f"**không nhất quán** với PhoBERT. Hãy xem xét cẩn thận!"
            )
    else:
        # BKAI corpus không có nhãn fake/real — chỉ corpus thủ tục
        lines.append("")
        lines.append("🟡 Corpus BKAI không có nhãn Fake/Real (các bài báo chính thuyết từ VnExpress).")
        lines.append("Kết quả của Web RAG (tab bên phải) cung cấp bằng chứng thực tế từ internet.")
        is_consistent = True  # Không mâu thuẫn vì không có nhãn đối chiếu

    if web_c > 0:
        lines.append(f"\n🌐 **{web_c}** bài từ Web RAG — xem tab '🌐 Web RAG' để kiểm tra nguồn thực tế.")

    return "\n".join(lines), is_consistent


def _why_fake_factors(label: str, conf: float, distribution: Dict[str, Any]) -> str:
    """Liệt kê các yếu tố gợi ý tại sao là fake/real."""
    factors = []

    if label == "FAKE":
        # Yếu tố phía mô hình
        if conf >= HIGH_CONF:
            factors.append("🔹 Mô hình PhoBERT phát hiện **mẫu ngôn ngữ đặc trưng của tin giả** với độ chắc chắn rất cao (ngôn ngữ cảm xúc, tuyên bố chưa có căn cứ, v.v.).")
        elif conf >= MEDIUM_CONF:
            factors.append("🔹 PhoBERT nhận diện **một số đặc điểm của tin giả** (ngôn ngữ thiếu trung lập, thông tin chưa được kiểm chứng).")
        else:
            factors.append("🔹 PhoBERT phân loại là Fake nhưng **độ tin cậy không cao** — kết quả có thể chưa chắc chắn.")

        # Yếu tố từ evidence
        fake_ratio = distribution["fake_count"] / distribution["total"] if distribution["total"] > 0 else 0
        if fake_ratio >= 0.6:
            factors.append(f"🔹 Phần lớn bài báo tương tự (**{fake_ratio:.0%}**) trong cơ sở dữ liệu cũng là **tin giả** — cho thấy chủ đề/cách viết tương đồng với các nguồn tin không đáng tin.")
        elif fake_ratio >= 0.4:
            factors.append(f"🔹 Có một tỉ lệ đáng kể bài tương tự (**{fake_ratio:.0%}**) là **tin giả** — cần thận trọng.")

        factors.append("🔹 Khuyến nghị: **Kiểm tra nguồn tin gốc**, tìm báo cáo từ các cơ quan báo chí uy tín trước khi chia sẻ.")

    else:  # REAL
        if conf >= HIGH_CONF:
            factors.append("🔹 Mô hình PhoBERT nhận diện **văn phong trung lập, khách quan** và **cấu trúc tin tức chuẩn mực** với độ chắc chắn rất cao.")
        elif conf >= MEDIUM_CONF:
            factors.append("🔹 PhoBERT nhận diện **phần lớn đặc điểm của tin thật** — ngôn ngữ cân bằng, có dẫn nguồn.")
        else:
            factors.append("🔹 PhoBERT phân loại là Real nhưng **độ tin cậy thấp** — kết quả không hoàn toàn chắc chắn.")

        real_ratio = distribution["real_count"] / distribution["total"] if distribution["total"] > 0 else 0
        if real_ratio >= 0.6:
            factors.append(f"🔹 Phần lớn bài báo tương tự (**{real_ratio:.0%}**) trong cơ sở dữ liệu là **tin thật** — nội dung phù hợp với các nguồn tin đáng tin cậy.")

        factors.append("🔹 Dù vậy, hãy luôn **xác minh thông tin** từ ít nhất 2-3 nguồn báo chí chính thống.")

    return "\n".join(factors)


def _disclaimer_text(label: str, conf: float) -> str:
    """Cảnh báo về giới hạn của mô hình."""
    base = (
        "\n---\n"
        "⚠️ **Lưu ý**: Đây là kết quả từ mô hình AI học máy (PhoBERT fine-tuned), "
        "**không phải kết luận cuối cùng từ chuyên gia kiểm chứng**. "
    )
    if conf < MEDIUM_CONF:
        base += (
            "Độ tin cậy thấp cho thấy mô hình **chưa chắc chắn** với văn bản này — "
            "hãy tự kiểm chứng kỹ hơn."
        )
    else:
        base += (
            "Luôn kiểm tra chéo với các nguồn tin chính thống như VnExpress, Tuổi Trẻ, "
            "Thanh Niên hoặc cơ quan chính phủ."
        )
    return base


# ─────────────────────────────────────────────────────────────
# Hàm chính
# ─────────────────────────────────────────────────────────────

def build_explanation(
    label: str,
    confidence: float,
    probabilities,
    evidences: List[Dict[str, Any]],
    distribution: Dict[str, Any],
) -> Dict[str, str]:
    """
    Tổng hợp toàn bộ giải thích thành một dict có cấu trúc.

    Returns:
        {
          "verdict":   câu kết luận chính,
          "evidence":  phân tích bằng chứng,
          "factors":   yếu tố quyết định,
          "disclaimer":cảnh báo,
          "is_consistent": bool — evidence có nhất quán với model không
        }
    """
    verdict  = _model_verdict_text(label, confidence)
    evidence_text, is_consistent = _evidence_analysis_text(distribution, label)
    factors  = _why_fake_factors(label, confidence, distribution)
    disclaimer = _disclaimer_text(label, confidence)

    return {
        "verdict":       verdict,
        "evidence":      evidence_text,
        "factors":       factors,
        "disclaimer":    disclaimer,
        "is_consistent": is_consistent,
    }
