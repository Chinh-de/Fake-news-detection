"use client";
import { useState } from "react";
import { newsApi } from "@/lib/api";

const SAMPLE_FAKE = "CHẤN ĐỘNG: Phát hiện vaccine COVID gây biến đổi gen người! Hàng nghìn người đã mọc đuôi sau khi tiêm. Chính phủ đang che giấu sự thật kinh hoàng này!";
const SAMPLE_REAL = "Ngày 19/5, Chính phủ họp phiên thường kỳ tháng 5, thảo luận về tình hình kinh tế - xã hội và các giải pháp thúc đẩy tăng trưởng GDP trong nửa cuối năm 2025.";

export default function PredictPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<"bm25" | "web">("bm25");
  const [settings, setSettings] = useState({
    max_length: 256,
    top_k_bm25: 4,
    top_k_web: 3,
    enable_web: true,
    crawl_web: false,
  });

  const handlePredict = async () => {
    if (text.trim().length < 10) {
      setError("Nội dung tin tức phải có ít nhất 10 ký tự");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await newsApi.predict({ text, ...settings });
      setResult(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Lỗi khi phân tích. Vui lòng thử lại.");
    } finally {
      setLoading(false);
    }
  };

  const isReal = result?.label === "REAL";

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🔍 Phân tích tin tức</h1>
        <p className="page-sub">Nhập nội dung tin tức để PhoBERT + RAG phân tích và đưa ra kết quả</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Input panel */}
        <div>
          <div className="card">
            <div className="section-header">📝 Nội dung tin tức</div>

            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem" }}>
              <button className="btn btn-ghost btn-sm" onClick={() => setText(SAMPLE_FAKE)}>📰 Mẫu tin giả</button>
              <button className="btn btn-ghost btn-sm" onClick={() => setText(SAMPLE_REAL)}>📰 Mẫu tin thật</button>
              <button className="btn btn-ghost btn-sm" onClick={() => { setText(""); setResult(null); }}>🗑️ Xóa</button>
            </div>

            {error && <div className="alert alert-error">{error}</div>}

            <textarea
              id="predict-text"
              className="form-textarea"
              style={{ minHeight: "180px" }}
              placeholder="Dán nội dung bài báo tiếng Việt vào đây...&#10;&#10;Ví dụ: Chính phủ vừa công bố quyết định tăng lương tối thiểu..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.3rem" }}>
              {text.length} ký tự
            </div>

            <button
              id="predict-submit"
              className="btn btn-primary btn-full btn-lg"
              style={{ marginTop: "1rem" }}
              onClick={handlePredict}
              disabled={loading}
            >
              {loading
                ? <><span className="spinner" /> Đang phân tích (PhoBERT + RAG)...</>
                : "🔍 Phân tích"}
            </button>
          </div>

          {/* Settings */}
          <div className="card" style={{ marginTop: "1rem" }}>
            <div className="section-header">⚙️ Cài đặt</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">BM25 top-k ({settings.top_k_bm25})</label>
                <input type="range" min={1} max={8} value={settings.top_k_bm25}
                  onChange={e => setSettings(s => ({ ...s, top_k_bm25: +e.target.value }))}
                  style={{ width: "100%", accentColor: "var(--accent)" }} />
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Web RAG top-k ({settings.top_k_web})</label>
                <input type="range" min={0} max={6} value={settings.top_k_web}
                  onChange={e => setSettings(s => ({ ...s, top_k_web: +e.target.value }))}
                  style={{ width: "100%", accentColor: "var(--accent)" }} />
              </div>
            </div>
            <div style={{ display: "flex", gap: "1.5rem", marginTop: "0.75rem" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", cursor: "pointer", fontSize: "0.85rem" }}>
                <input type="checkbox" checked={settings.enable_web}
                  onChange={e => setSettings(s => ({ ...s, enable_web: e.target.checked }))}
                  style={{ accentColor: "var(--accent)" }} />
                Bật Web RAG
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", cursor: "pointer", fontSize: "0.85rem" }}>
                <input type="checkbox" checked={settings.crawl_web}
                  onChange={e => setSettings(s => ({ ...s, crawl_web: e.target.checked }))}
                  style={{ accentColor: "var(--accent)" }} />
                Crawl trang web (chậm hơn)
              </label>
            </div>
          </div>
        </div>

        {/* Result panel */}
        <div>
          {!result && !loading && (
            <div style={{
              height: "220px",
              display: "flex", alignItems: "center", justifyContent: "center",
              flexDirection: "column", gap: "0.75rem",
              background: "rgba(30,41,59,0.4)",
              border: "2px dashed var(--border)",
              borderRadius: "var(--radius-lg)",
              color: "var(--text-muted)",
            }}>
              <span style={{ fontSize: "3rem" }}>🔍</span>
              <span>Nhập tin tức và bấm <strong>Phân tích</strong> để xem kết quả</span>
            </div>
          )}

          {loading && (
            <div style={{
              height: "220px",
              display: "flex", alignItems: "center", justifyContent: "center",
              flexDirection: "column", gap: "1rem",
              background: "rgba(59,130,246,0.05)",
              border: "1px solid rgba(59,130,246,0.2)",
              borderRadius: "var(--radius-lg)",
              color: "var(--text-secondary)",
            }}>
              <div className="spinner" style={{ width: "36px", height: "36px", borderWidth: "3px" }} />
              <div style={{ textAlign: "center" }}>
                <div style={{ fontWeight: 600 }}>🤖 Đang xử lý...</div>
                <div style={{ fontSize: "0.83rem", color: "var(--text-muted)", marginTop: "0.3rem" }}>
                  PhoBERT + BM25 Retrieval + Web RAG
                </div>
              </div>
            </div>
          )}

          {result && (
            <>
              {/* Verdict */}
              <div className={isReal ? "verdict-real" : "verdict-fake"}>
                <div className="verdict-label">
                  {isReal ? "🟢 TIN THẬT" : "🔴 TIN GIẢ"}
                </div>
                <div className="verdict-conf">
                  Độ tin cậy: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                  &nbsp;·&nbsp; Real: {(result.prob_real * 100).toFixed(1)}%
                  &nbsp;·&nbsp; Fake: {(result.prob_fake * 100).toFixed(1)}%
                </div>

                {/* Prob bars */}
                <div className="prob-bar-wrap">
                  <div className="prob-row">
                    <span className="prob-label-txt">🟢 REAL</span>
                    <div className="prob-bar">
                      <div className="prob-fill" style={{ width: `${result.prob_real * 100}%`, background: "var(--success)" }} />
                    </div>
                    <span className="prob-pct" style={{ color: "var(--success)" }}>{(result.prob_real * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prob-row">
                    <span className="prob-label-txt">🔴 FAKE</span>
                    <div className="prob-bar">
                      <div className="prob-fill" style={{ width: `${result.prob_fake * 100}%`, background: "var(--danger)" }} />
                    </div>
                    <span className="prob-pct" style={{ color: "var(--danger)" }}>{(result.prob_fake * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Explanation */}
              <div className="card" style={{ marginTop: "1rem" }}>
                <div className="section-header">💡 Giải thích</div>
                <p style={{ fontSize: "0.87rem", lineHeight: 1.7, color: "var(--text-secondary)" }}>
                  {result.explanation?.verdict?.replace(/\*\*/g, "")}
                </p>
                <div style={{ marginTop: "0.75rem", fontSize: "0.82rem", color: "var(--text-muted)" }}>
                  📚 Corpus: <strong>{result.corpus_name}</strong> ({(result.corpus_total || 0).toLocaleString()} bài)
                </div>
              </div>

              {/* Evidences */}
              <div className="card" style={{ marginTop: "1rem" }}>
                <div className="tabs">
                  <button className={`tab-btn ${activeTab === "bm25" ? "active" : ""}`}
                    onClick={() => setActiveTab("bm25")}>
                    📚 BM25 ({result.bm25_evidences?.length || 0})
                  </button>
                  <button className={`tab-btn ${activeTab === "web" ? "active" : ""}`}
                    onClick={() => setActiveTab("web")}>
                    🌐 Web RAG ({result.web_evidences?.length || 0})
                  </button>
                </div>

                {activeTab === "bm25" && (
                  <div>
                    {(result.bm25_evidences || []).length === 0 ? (
                      <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Không có bằng chứng BM25</p>
                    ) : (result.bm25_evidences || []).map((ev: any, i: number) => (
                      <div key={i} className="evidence-card" style={{ borderLeftColor: ev.label === 1 ? "var(--danger)" : "var(--success)" }}>
                        <div className="evidence-rank">#{ev.rank} &nbsp; BM25: {ev.score?.toFixed(3)}</div>
                        <div className="evidence-title">{(ev.title || "(Không có tiêu đề)").slice(0, 80)}</div>
                        <div className="evidence-snippet">{ev.snippet?.slice(0, 200)}</div>
                      </div>
                    ))}
                  </div>
                )}

                {activeTab === "web" && (
                  <div>
                    {!settings.enable_web ? (
                      <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Web RAG đang tắt</p>
                    ) : (result.web_evidences || []).length === 0 ? (
                      <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Không tìm được bằng chứng web</p>
                    ) : (result.web_evidences || []).map((ev: any, i: number) => (
                      <div key={i} className="evidence-card" style={{ borderLeftColor: "var(--accent)" }}>
                        <div className="evidence-rank">
                          #{ev.rank} &nbsp; 🌐 {ev.domain || "web"}
                          &nbsp; cosine: {ev.score?.toFixed(3)}
                          {ev.url && (
                            <a href={ev.url} target="_blank" rel="noopener noreferrer"
                              style={{ marginLeft: "0.5rem", color: "var(--accent)", fontSize: "0.7rem" }}>🔗</a>
                          )}
                        </div>
                        <div className="evidence-title">{(ev.title || "(Không có tiêu đề)").slice(0, 80)}</div>
                        <div className="evidence-snippet">{ev.snippet?.slice(0, 200)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
