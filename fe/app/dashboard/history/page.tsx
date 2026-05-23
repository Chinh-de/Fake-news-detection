"use client";
import { useEffect, useState } from "react";
import { newsApi } from "@/lib/api";

export default function HistoryPage() {
  const [submissions, setSubmissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [detail, setDetail] = useState<any>(null);

  useEffect(() => {
    newsApi.mySubmissions().then(setSubmissions).finally(() => setLoading(false));
  }, []);

  const filtered = filter === "all" ? submissions : submissions.filter(s => s.status === filter);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">📋 Lịch sử đóng góp</h1>
        <p className="page-sub">Danh sách tất cả tin tức bạn đã gửi và trạng thái xét duyệt</p>
      </div>

      {/* Filter tabs */}
      <div className="tabs">
        {["all", "pending", "approved", "rejected"].map(f => (
          <button key={f} className={`tab-btn ${filter === f ? "active" : ""}`} onClick={() => setFilter(f)}>
            {f === "all" ? "🗂️ Tất cả" : f === "pending" ? "⏳ Chờ duyệt" : f === "approved" ? "✅ Đã duyệt" : "❌ Từ chối"}
            &nbsp;({f === "all" ? submissions.length : submissions.filter(s => s.status === f).length})
          </button>
        ))}
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>
          <div className="spinner" style={{ margin: "0 auto 0.5rem", width: "32px", height: "32px" }} />
          Đang tải...
        </div>
      ) : filtered.length === 0 ? (
        <div style={{
          textAlign: "center", padding: "3rem",
          background: "var(--bg-glass)", border: "1px dashed var(--border)",
          borderRadius: "var(--radius-lg)", color: "var(--text-muted)",
        }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>📭</div>
          Không có tin nào trong danh mục này
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Tiêu đề</th>
                <th>Nhãn</th>
                <th>Trạng thái</th>
                <th>Ghi chú admin</th>
                <th>Ngày gửi</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => (
                <tr key={s.id}>
                  <td style={{ color: "var(--text-muted)" }}>{i + 1}</td>
                  <td style={{ maxWidth: "260px" }}>
                    <div className="truncate" style={{ color: "var(--text-primary)", fontWeight: 500 }}>{s.title}</div>
                    <div className="text-xs text-muted truncate">{s.content.slice(0, 80)}...</div>
                  </td>
                  <td><span className={`badge badge-${s.user_label.toLowerCase()}`}>{s.user_label}</span></td>
                  <td><span className={`badge badge-${s.status}`}>{s.status}</span></td>
                  <td style={{ maxWidth: "180px" }}>
                    <span className="text-xs text-secondary truncate" style={{ display: "block" }}>
                      {s.admin_note || "—"}
                    </span>
                  </td>
                  <td className="text-muted text-sm">{new Date(s.created_at).toLocaleDateString("vi-VN")}</td>
                  <td>
                    <button className="btn btn-ghost btn-sm" onClick={() => setDetail(s)}>👁️ Xem</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Detail modal */}
      {detail && (
        <div className="modal-overlay" onClick={() => setDetail(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="modal-title">📄 Chi tiết đóng góp #{detail.id}</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setDetail(null)}>✕</button>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <div className="form-label">Tiêu đề</div>
              <div style={{ color: "var(--text-primary)", fontWeight: 500 }}>{detail.title}</div>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <div className="form-label">Nhãn kiểm chứng</div>
              <span className={`badge badge-${detail.user_label.toLowerCase()}`}>{detail.user_label}</span>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <div className="form-label">Trạng thái</div>
              <span className={`badge badge-${detail.status}`}>{detail.status}</span>
            </div>

            {detail.source_url && (
              <div style={{ marginBottom: "1rem" }}>
                <div className="form-label">URL nguồn</div>
                <a href={detail.source_url} target="_blank" rel="noopener noreferrer"
                  style={{ color: "var(--accent)", fontSize: "0.85rem", wordBreak: "break-all" }}>
                  {detail.source_url}
                </a>
              </div>
            )}

            {detail.admin_note && (
              <div style={{ marginBottom: "1rem" }}>
                <div className="form-label">Ghi chú từ admin</div>
                <div className="alert alert-info" style={{ margin: 0 }}>{detail.admin_note}</div>
              </div>
            )}

            <div style={{ marginBottom: "1rem" }}>
              <div className="form-label">Nội dung</div>
              <div style={{
                background: "rgba(0,0,0,0.3)", borderRadius: "var(--radius-sm)",
                padding: "0.75rem", fontSize: "0.85rem", lineHeight: 1.7,
                color: "var(--text-secondary)", maxHeight: "200px", overflowY: "auto",
              }}>
                {detail.content}
              </div>
            </div>

            <div className="text-xs text-muted">
              Ngày gửi: {new Date(detail.created_at).toLocaleString("vi-VN")}
              {detail.reviewed_at && ` · Xét duyệt: ${new Date(detail.reviewed_at).toLocaleString("vi-VN")}`}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
