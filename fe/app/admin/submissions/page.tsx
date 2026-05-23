"use client";
import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { adminApi } from "@/lib/api";

function SubmissionsContent() {
  const searchParams = useSearchParams();
  const [submissions, setSubmissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState(searchParams.get("status") || "all");
  const [detail, setDetail] = useState<any>(null);
  const [noteInput, setNoteInput] = useState("");
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  const loadData = () => {
    setLoading(true);
    adminApi.submissions(filter === "all" ? undefined : filter)
      .then(setSubmissions)
      .finally(() => setLoading(false));
  };

  useEffect(loadData, [filter]);

  const handleApprove = async (id: number) => {
    setActionLoading(id);
    try {
      await adminApi.approve(id, noteInput || undefined);
      setDetail(null);
      setNoteInput("");
      loadData();
    } catch (err: any) { alert(err.response?.data?.detail || "Lỗi"); }
    finally { setActionLoading(null); }
  };

  const handleReject = async (id: number) => {
    if (!noteInput.trim()) { alert("Vui lòng nhập lý do từ chối"); return; }
    setActionLoading(id);
    try {
      await adminApi.reject(id, noteInput);
      setDetail(null);
      setNoteInput("");
      loadData();
    } catch (err: any) { alert(err.response?.data?.detail || "Lỗi"); }
    finally { setActionLoading(null); }
  };

  const approvedSelected = submissions.filter(s => selectedIds.includes(s.id) && s.status === "approved");

  return (
    <div>
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="page-title">📥 Duyệt Submissions</h1>
            <p className="page-sub">Xem xét và phê duyệt tin tức từ cộng đồng để đưa vào retrain model</p>
          </div>
          {approvedSelected.length > 0 && (
            <a href="/admin/retrain" className="btn btn-primary">
              🤖 Retrain với {approvedSelected.length} tin đã chọn →
            </a>
          )}
        </div>
      </div>

      {/* Filter tabs */}
      <div className="tabs">
        {["all", "pending", "approved", "rejected"].map(f => (
          <button key={f} className={`tab-btn ${filter === f ? "active" : ""}`} onClick={() => { setFilter(f); setSelectedIds([]); }}>
            {f === "all" ? "🗂️ Tất cả" : f === "pending" ? "⏳ Chờ duyệt" : f === "approved" ? "✅ Đã duyệt" : "❌ Từ chối"}
          </button>
        ))}
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>
          <div className="spinner" style={{ margin: "0 auto 0.5rem", width: "32px", height: "32px" }} /> Đang tải...
        </div>
      ) : submissions.length === 0 ? (
        <div style={{
          textAlign: "center", padding: "3rem",
          background: "var(--bg-glass)", border: "1px dashed var(--border)",
          borderRadius: "var(--radius-lg)", color: "var(--text-muted)",
        }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>📭</div>
          Không có submission nào
        </div>
      ) : (
        <div className="card" style={{ padding: 0 }}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th style={{ width: "40px" }}>
                    <input type="checkbox"
                      onChange={e => setSelectedIds(e.target.checked ? submissions.map(s => s.id) : [])}
                      checked={selectedIds.length === submissions.length && submissions.length > 0}
                      style={{ accentColor: "var(--accent)" }} />
                  </th>
                  <th>#</th>
                  <th>Tiêu đề & Nội dung</th>
                  <th>Người gửi</th>
                  <th>Nhãn</th>
                  <th>Trạng thái</th>
                  <th>Ngày</th>
                  <th>Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {submissions.map(s => (
                  <tr key={s.id} style={{ background: selectedIds.includes(s.id) ? "rgba(59,130,246,0.05)" : undefined }}>
                    <td>
                      <input type="checkbox" checked={selectedIds.includes(s.id)}
                        onChange={e => setSelectedIds(prev => e.target.checked ? [...prev, s.id] : prev.filter(id => id !== s.id))}
                        style={{ accentColor: "var(--accent)" }} />
                    </td>
                    <td className="text-muted">#{s.id}</td>
                    <td style={{ maxWidth: "260px" }}>
                      <div className="truncate" style={{ fontWeight: 500, color: "var(--text-primary)" }}>{s.title}</div>
                      <div className="text-xs text-muted truncate">{s.content?.slice(0, 60)}...</div>
                    </td>
                    <td className="text-secondary text-sm">{s.author?.username || `#${s.user_id}`}</td>
                    <td><span className={`badge badge-${s.user_label.toLowerCase()}`}>{s.user_label}</span></td>
                    <td><span className={`badge badge-${s.status}`}>{s.status}</span></td>
                    <td className="text-muted text-sm">{new Date(s.created_at).toLocaleDateString("vi-VN")}</td>
                    <td>
                      <div style={{ display: "flex", gap: "0.4rem" }}>
                        <button className="btn btn-ghost btn-sm"
                          onClick={() => { setDetail(s); setNoteInput(s.admin_note || ""); }}>
                          👁️ Xem
                        </button>
                        {s.status === "pending" && (
                          <>
                            <button className="btn btn-success btn-sm"
                              disabled={actionLoading === s.id}
                              onClick={async () => { setActionLoading(s.id); await adminApi.approve(s.id); loadData(); setActionLoading(null); }}>
                              ✅
                            </button>
                            <button className="btn btn-danger btn-sm"
                              disabled={actionLoading === s.id}
                              onClick={() => { setDetail(s); setNoteInput(""); }}>
                              ❌
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Detail / Review modal */}
      {detail && (
        <div className="modal-overlay" onClick={() => setDetail(null)}>
          <div className="modal-box" style={{ maxWidth: "600px" }} onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="modal-title">📄 Submission #{detail.id}</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setDetail(null)}>✕</button>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
              <div>
                <div className="form-label">Người gửi</div>
                <div style={{ fontWeight: 500 }}>{detail.author?.username || `User #${detail.user_id}`}</div>
              </div>
              <div>
                <div className="form-label">Nhãn kiểm chứng</div>
                <span className={`badge badge-${detail.user_label.toLowerCase()}`}>{detail.user_label}</span>
              </div>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <div className="form-label">Tiêu đề</div>
              <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{detail.title}</div>
            </div>

            {detail.source_url && (
              <div style={{ marginBottom: "1rem" }}>
                <div className="form-label">URL nguồn</div>
                <a href={detail.source_url} target="_blank" rel="noopener noreferrer"
                  style={{ color: "var(--accent)", fontSize: "0.83rem", wordBreak: "break-all" }}>
                  🔗 {detail.source_url}
                </a>
              </div>
            )}

            <div style={{ marginBottom: "1.2rem" }}>
              <div className="form-label">Nội dung</div>
              <div style={{
                background: "rgba(0,0,0,0.4)", borderRadius: "var(--radius-sm)",
                padding: "0.75rem", fontSize: "0.85rem", lineHeight: 1.7,
                color: "var(--text-secondary)", maxHeight: "200px", overflowY: "auto",
                border: "1px solid var(--border)",
              }}>
                {detail.content}
              </div>
            </div>

            {detail.status === "pending" && (
              <>
                <div className="form-group">
                  <label className="form-label">Ghi chú admin (bắt buộc khi từ chối)</label>
                  <textarea className="form-textarea" style={{ minHeight: "80px" }}
                    placeholder="Nhập ghi chú, lý do duyệt/từ chối..."
                    value={noteInput}
                    onChange={e => setNoteInput(e.target.value)} />
                </div>
                <div style={{ display: "flex", gap: "0.75rem" }}>
                  <button className="btn btn-success" style={{ flex: 1 }}
                    disabled={actionLoading === detail.id}
                    onClick={() => handleApprove(detail.id)}>
                    {actionLoading === detail.id ? <><span className="spinner" /> Đang xử lý...</> : "✅ Phê duyệt"}
                  </button>
                  <button className="btn btn-danger" style={{ flex: 1 }}
                    disabled={actionLoading === detail.id}
                    onClick={() => handleReject(detail.id)}>
                    {actionLoading === detail.id ? <><span className="spinner" /> Đang xử lý...</> : "❌ Từ chối"}
                  </button>
                  <button className="btn btn-ghost" onClick={() => setDetail(null)}>Đóng</button>
                </div>
              </>
            )}

            {detail.status !== "pending" && (
              <div>
                {detail.admin_note && (
                  <div className="alert alert-info">📝 Ghi chú: {detail.admin_note}</div>
                )}
                <button className="btn btn-ghost btn-full" onClick={() => setDetail(null)}>Đóng</button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AdminSubmissionsPage() {
  return (
    <Suspense fallback={<div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}><div className="spinner" /></div>}>
      <SubmissionsContent />
    </Suspense>
  );
}
