"use client";
import { useState } from "react";
import { newsApi } from "@/lib/api";

export default function SubmitPage() {
  const [form, setForm] = useState({
    title: "",
    content: "",
    user_label: "FAKE",
    source_url: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess(false);
    try {
      await newsApi.submit({
        ...form,
        source_url: form.source_url || undefined,
      });
      setSuccess(true);
      setForm({ title: "", content: "", user_label: "FAKE", source_url: "" });
    } catch (err: any) {
      setError(err.response?.data?.detail || "Gửi thất bại. Vui lòng thử lại.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">📤 Đóng góp tin kiểm chứng</h1>
        <p className="page-sub">
          Gửi tin tức đã được bạn kiểm chứng thực tế — admin sẽ xem xét và dùng để cải thiện mô hình AI
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "1.5rem" }}>
        <div className="card">
          {error && <div className="alert alert-error">{error}</div>}
          {success && (
            <div className="alert alert-success">
              ✅ Tin đã được gửi thành công! Admin sẽ xem xét trong thời gian sớm nhất.
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Tiêu đề tin tức *</label>
              <input
                id="submit-title"
                className="form-input"
                type="text"
                placeholder="Tiêu đề đầy đủ của bài báo..."
                value={form.title}
                onChange={e => setForm({ ...form, title: e.target.value })}
                required minLength={5}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Nội dung tin tức *</label>
              <textarea
                id="submit-content"
                className="form-textarea"
                style={{ minHeight: "200px" }}
                placeholder="Nội dung đầy đủ của tin tức (ít nhất 20 ký tự)..."
                value={form.content}
                onChange={e => setForm({ ...form, content: e.target.value })}
                required minLength={20}
              />
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                {form.content.length} ký tự
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Nhãn kiểm chứng *</label>
              <div style={{ display: "flex", gap: "1rem" }}>
                <label style={{
                  flex: 1, cursor: "pointer",
                  padding: "0.85rem 1rem",
                  borderRadius: "var(--radius-sm)",
                  border: `2px solid ${form.user_label === "FAKE" ? "var(--danger)" : "var(--border)"}`,
                  background: form.user_label === "FAKE" ? "rgba(239,68,68,0.08)" : "transparent",
                  display: "flex", alignItems: "center", gap: "0.5rem",
                  transition: "all 0.2s",
                }}>
                  <input type="radio" name="label" value="FAKE"
                    checked={form.user_label === "FAKE"}
                    onChange={() => setForm({ ...form, user_label: "FAKE" })}
                    style={{ accentColor: "var(--danger)" }} />
                  <span>🔴 <strong>Tin GIẢ</strong></span>
                  <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>— thông tin sai lệch</span>
                </label>
                <label style={{
                  flex: 1, cursor: "pointer",
                  padding: "0.85rem 1rem",
                  borderRadius: "var(--radius-sm)",
                  border: `2px solid ${form.user_label === "REAL" ? "var(--success)" : "var(--border)"}`,
                  background: form.user_label === "REAL" ? "rgba(16,185,129,0.08)" : "transparent",
                  display: "flex", alignItems: "center", gap: "0.5rem",
                  transition: "all 0.2s",
                }}>
                  <input type="radio" name="label" value="REAL"
                    checked={form.user_label === "REAL"}
                    onChange={() => setForm({ ...form, user_label: "REAL" })}
                    style={{ accentColor: "var(--success)" }} />
                  <span>🟢 <strong>Tin THẬT</strong></span>
                  <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>— đã được kiểm chứng</span>
                </label>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">URL nguồn (không bắt buộc)</label>
              <input
                id="submit-url"
                className="form-input"
                type="url"
                placeholder="https://tuoitre.vn/bai-bao-goc..."
                value={form.source_url}
                onChange={e => setForm({ ...form, source_url: e.target.value })}
              />
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Cung cấp link nguồn giúp admin kiểm chứng nhanh hơn
              </div>
            </div>

            <button
              id="submit-btn"
              type="submit"
              className="btn btn-primary btn-full"
              style={{ padding: "0.75rem", fontSize: "0.95rem" }}
              disabled={loading}
            >
              {loading ? <><span className="spinner" /> Đang gửi...</> : "📤 Gửi đóng góp"}
            </button>
          </form>
        </div>

        {/* Info sidebar */}
        <div>
          <div className="card">
            <div className="section-header">📌 Hướng dẫn</div>
            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8 }}>
              <div style={{ marginBottom: "1rem" }}>
                <strong style={{ color: "var(--text-primary)" }}>1. Kiểm chứng kỹ trước khi gửi</strong>
                <p>Chỉ gửi tin mà bạn đã xác minh thực tế từ nhiều nguồn đáng tin cậy.</p>
              </div>
              <div style={{ marginBottom: "1rem" }}>
                <strong style={{ color: "var(--text-primary)" }}>2. Điền đầy đủ thông tin</strong>
                <p>Tiêu đề, nội dung và URL nguồn giúp admin xét duyệt nhanh hơn.</p>
              </div>
              <div style={{ marginBottom: "1rem" }}>
                <strong style={{ color: "var(--text-primary)" }}>3. Đánh nhãn chính xác</strong>
                <p>Nhãn sai lệch có thể làm giảm chất lượng mô hình AI.</p>
              </div>
              <div>
                <strong style={{ color: "var(--text-primary)" }}>4. Quy trình duyệt</strong>
                <p>Admin sẽ xem xét và phê duyệt. Tin được duyệt sẽ dùng để retrain mô hình.</p>
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: "1rem" }}>
            <div className="section-header">⏱️ Trạng thái quy trình</div>
            <div style={{ fontSize: "0.82rem" }}>
              {[
                { step: "Bạn gửi tin", status: "pending", desc: "Chờ admin xem xét" },
                { step: "Admin duyệt", status: "approved", desc: "Tin được chấp nhận" },
                { step: "Đưa vào retrain", status: "done", desc: "Cải thiện mô hình AI" },
              ].map((s, i) => (
                <div key={i} style={{ display: "flex", gap: "0.75rem", marginBottom: "0.75rem", alignItems: "flex-start" }}>
                  <div style={{
                    width: "24px", height: "24px", borderRadius: "50%",
                    background: "var(--grad-accent)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: "0.7rem", fontWeight: 800, flexShrink: 0,
                    color: "#fff",
                  }}>{i + 1}</div>
                  <div>
                    <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{s.step}</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>{s.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
