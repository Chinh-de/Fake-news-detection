# app.py — Vietnamese Fake News Detector with Retrieval-Augmented Explanation
# ============================================================================
# Nâng cấp từ app.py gốc:
#  • Thêm BM25 retrieval từ corpus ViFN train_cleaned.csv
#  • Giải thích tại sao tin là THẬT / GIẢ (Explainability)
#  • Hiển thị top-k bằng chứng (evidence) tương tự
#  • Giao diện đẹp, responsive, với màu sắc và icon trực quan

import re
import sys
import io

import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─────────────────────────────────────────────────────────────
# Fix stdout encoding (Windows terminal UTF-8)
# ─────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# Import modules
# ─────────────────────────────────────────────────────────────
from retriever import retrieve_evidence, analyze_retrieved_distribution
from explainer import build_explanation

# ─────────────────────────────────────────────────────────────
# Cấu hình trang
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ViFN Fake News Detector — PhoBERT + Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS tùy chỉnh (dark mode + glassmorphism)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        border: 1px solid rgba(99,179,237,0.15);
    }
    .hero-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #63b3ed, #90cdf4, #bee3f8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; padding: 0;
    }
    .hero-sub {
        color: #94a3b8; font-size: 0.95rem; margin-top: 0.4rem;
    }

    /* Verdict cards */
    .verdict-fake {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #f87171;
        border-radius: 14px; padding: 1.4rem 1.8rem;
        box-shadow: 0 4px 20px rgba(239,68,68,0.3);
    }
    .verdict-real {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #34d399;
        border-radius: 14px; padding: 1.4rem 1.8rem;
        box-shadow: 0 4px 20px rgba(52,211,153,0.3);
    }
    .verdict-text {
        font-size: 1.5rem; font-weight: 700; color: #f8fafc;
    }
    .conf-text { color: #cbd5e1; font-size: 0.9rem; margin-top: 0.3rem; }

    /* Evidence cards */
    .evidence-card-fake {
        background: rgba(127,29,29,0.25);
        border-left: 4px solid #f87171;
        border-radius: 8px; padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .evidence-card-real {
        background: rgba(6,78,59,0.25);
        border-left: 4px solid #34d399;
        border-radius: 8px; padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .evidence-rank {
        font-weight: 700; font-size: 0.8rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .evidence-title {
        font-weight: 600; font-size: 0.95rem; color: #e2e8f0;
        margin: 0.25rem 0;
    }
    .evidence-snippet { color: #94a3b8; font-size: 0.85rem; line-height: 1.6; }
    .score-badge {
        display: inline-block; background: rgba(99,179,237,0.15);
        color: #63b3ed; border-radius: 9999px;
        padding: 0.1rem 0.55rem; font-size: 0.75rem; font-weight: 600;
        margin-left: 0.4rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem 0;
    }

    /* Info box */
    .info-box {
        background: rgba(30,58,138,0.2);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 10px; padding: 1rem 1.2rem;
        color: #cbd5e1; font-size: 0.88rem; line-height: 1.7;
    }

    /* Probability bar background */
    .prob-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🔍 ViFN Fake News Detector</div>
        <div class="hero-sub">
            Phát hiện tin giả tiếng Việt bằng <b>PhoBERT</b> + <b>Retrieval-Augmented Explanation</b>
            · Dữ liệu: ViFN Dataset · Model: fine-tuned PhoBERT-base
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Sidebar — Cấu hình
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")
    top_k_bm25 = st.slider(
        "Bằng chứng corpus (BM25 top-k)", min_value=1, max_value=8, value=4
    )
    top_k_web = st.slider(
        "Bằng chứng web (RAG top-k)", min_value=0, max_value=6, value=3
    )
    enable_web = st.toggle("Bật Web RAG (DDG search)", value=True)
    crawl_web = st.toggle("Crawl nội dung trang", value=True,
                           help="Lấy nội dung đầy đủ từ trang báo; chậm hơn nhưng chất lượng hơn")
    max_length = st.slider(
        "Max token (PhoBERT)", min_value=64, max_value=256, value=256, step=32
    )
    st.markdown("---")
    st.markdown(
        """
        **Về hệ thống:**
        - 🤖 **SLM**: PhoBERT-base (fine-tuned ViFN)
        - 📚 **Corpus**: BKAI NewsCategory (596K bài VnExpress)
        - 🔎 **BM25**: rank_bm25 (local search)
        - 🌐 **Web RAG**: DuckDuckGo → báo Việt
        - 💡 **Giải thích**: Rule-based + Evidence
        """
    )
    st.markdown("---")
    st.caption("PBL7 · Vietnamese Fake News Detection")
    top_k = top_k_bm25  # backward compat


# ─────────────────────────────────────────────────────────────
# Load PhoBERT model (cache 1 lần)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "./model"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


with st.spinner("⏳ Đang tải PhoBERT model..."):
    tokenizer, model, load_err = load_model()

if load_err or tokenizer is None:
    st.error(f"❌ Không thể tải model: {load_err}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# PhoBERT inference
# ─────────────────────────────────────────────────────────────
def predict(text: str, max_len: int = 256):
    """Chạy PhoBERT inference, trả về (label_str, confidence, probabilities)."""
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy().flatten()

    pred_id = int(np.argmax(probs))
    label_map = {0: "REAL", 1: "FAKE"}
    return label_map[pred_id], float(probs[pred_id]), probs


# ─────────────────────────────────────────────────────────────
# Clean input text (nhẹ, giữ tiếng Việt)
# ─────────────────────────────────────────────────────────────
def light_clean(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
# ── Mẫu nhanh — phải khởi tạo TRƯỚC khi tạo widget ──
SAMPLE_FAKE = (
    "CHẤN ĐỘNG: Phát hiện vaccine COVID gây biến đổi gen người! "
    "Hàng nghìn người đã mọc đuôi sau khi tiêm. Chính phủ đang che giấu sự thật kinh hoàng này!"
)
SAMPLE_REAL = (
    "Ngày 19/5, Chính phủ họp phiên thường kỳ tháng 5, thảo luận về tình hình "
    "kinh tế - xã hội và các giải pháp thúc đẩy tăng trưởng GDP trong nửa cuối năm 2025."
)

# Xử lý nút mẫu nhanh (phải chạy TRƯỚC khi khởi tạo text_area)
if st.session_state.get("_load_sample") == "fake":
    st.session_state["_default_text"] = SAMPLE_FAKE
    st.session_state.pop("_load_sample")
elif st.session_state.get("_load_sample") == "real":
    st.session_state["_default_text"] = SAMPLE_REAL
    st.session_state.pop("_load_sample")
elif st.session_state.get("_load_sample") == "clear":
    st.session_state["_default_text"] = ""
    st.session_state.pop("_load_sample")

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="section-header">📝 Nhập nội dung tin tức</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        label="Nội dung tin tức",
        label_visibility="collapsed",
        height=240,
        placeholder="Dán nội dung bài báo tiếng Việt vào đây...\n\nVí dụ: Chính phủ vừa công bố quyết định tăng lương tối thiểu...",
        value=st.session_state.get("_default_text", ""),
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        analyze = st.button("🔍 Phân tích", type="primary", use_container_width=True)
    with btn_col2:
        clear = st.button("🗑️ Xóa", use_container_width=True)

    if clear:
        st.session_state["_load_sample"] = "clear"
        st.rerun()

    # Ví dụ mẫu nhanh
    st.markdown("**🚀 Thử nhanh với mẫu:**")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button("📰 Ví dụ tin giả", use_container_width=True):
            st.session_state["_load_sample"] = "fake"
            st.rerun()
    with ex_col2:
        if st.button("📰 Ví dụ tin thật", use_container_width=True):
            st.session_state["_load_sample"] = "real"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Phân tích khi bấm nút
# ─────────────────────────────────────────────────────────────
if analyze and user_text and len(user_text.strip()) > 10:
    cleaned_text = light_clean(user_text)

    with st.spinner("🤖 Phân tích (PhoBERT + Retrieval)..."):
        # 1. PhoBERT prediction
        label, conf, probs = predict(cleaned_text, max_len=max_length)

        # 2. Dual-branch retrieval (BM25 + Web RAG)
        retrieval_result = retrieve_evidence(
            cleaned_text,
            top_k_bm25=top_k_bm25,
            top_k_web=top_k_web,
            enable_web=enable_web,
            crawl_web=crawl_web,
        )
        bm25_evs  = retrieval_result["bm25"]
        web_evs   = retrieval_result["web"]
        all_evs   = retrieval_result["all"]
        corpus_name = retrieval_result["corpus"]
        corpus_total = retrieval_result["corpus_total"]

        # 3. Distribution & explanation
        distribution = analyze_retrieved_distribution(retrieval_result)
        explanation  = build_explanation(label, conf, probs, all_evs, distribution)

    # ─── Cột kết quả ───
    with col_result:
        # Verdict card
        card_cls = "verdict-fake" if label == "FAKE" else "verdict-real"
        icon = "🔴" if label == "FAKE" else "🟢"
        label_vn = "TIN GIẢ" if label == "FAKE" else "TIN THẬT"
        st.markdown(
            f"""
            <div class="{card_cls}">
                <div class="verdict-text">{icon} {label_vn}</div>
                <div class="conf-text">Độ tin cậy: <b>{conf:.1%}</b> &nbsp;|&nbsp; 
                Real: {probs[0]:.1%} &nbsp;|&nbsp; Fake: {probs[1]:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bars
        st.markdown('<p class="prob-label" style="margin-top:0.8rem">📊 Xác suất phân loại</p>', unsafe_allow_html=True)
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("🟢 REAL", f"{probs[0]:.1%}")
            st.progress(float(probs[0]))
        with prob_col2:
            st.metric("🔴 FAKE", f"{probs[1]:.1%}")
            st.progress(float(probs[1]))

    # ─── Giải thích (full width dưới) ───
    st.markdown("---")
    exp_col1, exp_col2 = st.columns([1, 1], gap="large")

    with exp_col1:
        st.markdown('<div class="section-header">💡 Giải thích kết quả</div>', unsafe_allow_html=True)
        st.markdown(explanation["verdict"])

        # Corpus info
        st.caption(f"📚 Corpus: **{corpus_name}** ({corpus_total:,} bài) | Web RAG: **{'Bật' if enable_web else 'Tắt'}**")

        st.markdown('<div class="section-header">🔎 Phân tích bằng chứng</div>', unsafe_allow_html=True)
        is_consistent = explanation["is_consistent"]
        badge = "✅ Nhất quán" if is_consistent else "⚡ Mâu thuẫn"
        st.markdown(f"**Trạng thái:** {badge}")
        st.markdown(explanation["evidence"])

        st.markdown('<div class="section-header">📋 Các yếu tố quyết định</div>', unsafe_allow_html=True)
        st.markdown(explanation["factors"])

        st.markdown(
            f'<div class="info-box">{explanation["disclaimer"]}</div>',
            unsafe_allow_html=True,
        )

    with exp_col2:
        # Tab: BM25 vs Web
        tab_bm25, tab_web = st.tabs([
            f"📚 Corpus BKAI ({len(bm25_evs)})",
            f"🌐 Web RAG ({len(web_evs)})",
        ])

        # ── Tab BM25 ──
        with tab_bm25:
            if not bm25_evs:
                st.info("⚠️ Đang tải corpus BKAI lần đầu (có thể mất vài giây)...")
            else:
                st.caption(f"Bài báo tương đồng từ corpus **{corpus_name}**")
                for ev in bm25_evs:
                    # Màu card: BKAI không có nhãn fake/real → dùng màu trung tính
                    if ev.get("label") == 1:
                        card_cls, ev_icon, ev_lbl = "evidence-card-fake", "🔴", "Tin giả"
                    elif ev.get("label") == 0:
                        card_cls, ev_icon, ev_lbl = "evidence-card-real", "🟢", "Tin thật"
                    else:
                        card_cls, ev_icon, ev_lbl = "evidence-card-real", "🟡", ev.get("category", "Bài báo")

                    title_disp = (ev["title"] or "(Không có tiêu đề)")[:80]
                    st.markdown(
                        f"""
                        <div class="{card_cls}">
                            <div class="evidence-rank">
                                #{ev['rank']} &nbsp; {ev_icon} {ev_lbl}
                                <span class="score-badge">BM25: {ev['score']:.3f}</span>
                            </div>
                            <div class="evidence-title">{title_disp}</div>
                            <div class="evidence-snippet">{ev['snippet_highlighted']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Tab Web RAG ──
        with tab_web:
            if not enable_web:
                st.info("🔕 Web RAG đang tắt. Bật trong sidebar.")
            elif not web_evs:
                st.warning("⚠️ Không tìm được bằng chứng web. Kiểm tra kết nối mạng.")
            else:
                # Hiển thị các query đã dùng để search
                queries_used = retrieval_result.get("queries_used", [])
                if queries_used:
                    with st.expander("🔍 Queries đã dùng để tìm kiếm", expanded=False):
                        for q_str, q_type in queries_used:
                            q_type_map = {
                                "entity_stats": "🏷️ Thực thể + Số liệu",
                                "claim_sentence": "📌 Câu claim chính",
                                "broad_nouns": "🔎 Từ khóa rộng",
                                "fallback": "⚠️ Fallback",
                            }
                            label = q_type_map.get(q_type, q_type)
                            st.markdown(f"`{label}`: *{q_str[:100]}*")

                st.caption(f"Bằng chứng từ các báo Việt uy tín • Score = TF-IDF cosine similarity")
                for ev in web_evs:
                    domain = ev.get("domain", "")
                    url = ev.get("url", "")
                    q_type = ev.get("query_type", "")
                    title_disp = (ev["title"] or "(Không có tiêu đề)")[:80]
                    link_html = f'<a href="{url}" target="_blank" style="color:#63b3ed;font-size:0.75rem">🔗 {domain}</a>' if url else ""
                    q_badge = f'<span style="font-size:0.7rem;color:#a78bfa;margin-left:6px">[{q_type}]</span>' if q_type else ""
                    score_pct = min(int(ev['score'] * 100 / 0.5 * 100), 100)  # normalize to 100%
                    st.markdown(
                        f"""
                        <div class="evidence-card-real" style="border-left-color:#63b3ed">
                            <div class="evidence-rank">
                                #{ev['rank']} &nbsp; 🌐 Web
                                <span class="score-badge">cosine: {ev['score']:.3f}</span>
                                &nbsp; {link_html}{q_badge}
                            </div>
                            <div class="evidence-title">{title_disp}</div>
                            <div class="evidence-snippet">{ev['snippet_highlighted']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

elif analyze:
    st.warning("⚠️ Vui lòng nhập nội dung tin tức (tối thiểu 10 ký tự).")
else:
    # Placeholder khi chưa phân tích
    with col_result:
        st.markdown(
            """
            <div style="
                height: 240px; display: flex; align-items: center; justify-content: center;
                background: rgba(30,41,59,0.5); border-radius: 12px;
                border: 2px dashed #334155; color: #475569; font-size: 0.95rem;
                flex-direction: column; gap: 0.5rem;
            ">
                <span style="font-size:2.5rem">🔍</span>
                <span>Nhập tin tức và bấm <b>Phân tích</b> để xem kết quả</span>
            </div>
            """,
            unsafe_allow_html=True,
        )