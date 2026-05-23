"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { newsApi } from "@/lib/api";

export default function UserDashboard() {
  const [user, setUser] = useState<any>(null);
  const [submissions, setSubmissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const u = localStorage.getItem("vifn_user");
    if (u) setUser(JSON.parse(u));
    newsApi.mySubmissions().then(setSubmissions).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const pending = submissions.filter(s => s.status === "pending").length;
  const approved = submissions.filter(s => s.status === "approved").length;
  const rejected = submissions.filter(s => s.status === "rejected").length;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Xin chào, {user?.username || "bạn"} 👋</h1>
        <p className="page-sub">Nền tảng phát hiện tin giả tiếng Việt — PhoBERT + RAG</p>
      </div>

      {/* Quick actions */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "2rem" }}>
        <Link href="/dashboard/predict" style={{ textDecoration: "none" }}>
          <div className="card" style={{
            background: "linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.1))",
            border: "1px solid rgba(59,130,246,0.3)",
            cursor: "pointer",
            transition: "transform 0.2s, box-shadow 0.2s",
          }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = "translateY(-3px)"; (e.currentTarget as HTMLElement).style.boxShadow = "0 8px 30px rgba(59,130,246,0.2)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = ""; (e.currentTarget as HTMLElement).style.boxShadow = ""; }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>🔍</div>
            <div style={{ fontWeight: 700, fontSize: "1.05rem", marginBottom: "0.3rem" }}>Phân tích tin tức</div>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
              Nhập nội dung tin và để PhoBERT + RAG phân tích ngay
            </p>
            <div style={{ marginTop: "1rem" }}>
              <span className="btn btn-primary btn-sm">Phân tích ngay →</span>
            </div>
          </div>
        </Link>

        <Link href="/dashboard/submit" style={{ textDecoration: "none" }}>
          <div className="card" style={{
            background: "linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,78,59,0.15))",
            border: "1px solid rgba(16,185,129,0.25)",
            cursor: "pointer",
            transition: "transform 0.2s, box-shadow 0.2s",
          }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = "translateY(-3px)"; (e.currentTarget as HTMLElement).style.boxShadow = "0 8px 30px rgba(16,185,129,0.15)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = ""; (e.currentTarget as HTMLElement).style.boxShadow = ""; }}
          >
            <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>📤</div>
            <div style={{ fontWeight: 700, fontSize: "1.05rem", marginBottom: "0.3rem" }}>Đóng góp tin kiểm chứng</div>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
              Gửi tin tức đã kiểm chứng để giúp cải thiện mô hình AI
            </p>
            <div style={{ marginTop: "1rem" }}>
              <span className="btn btn-success btn-sm">Đóng góp →</span>
            </div>
          </div>
        </Link>
      </div>

      {/* Stats */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Tổng đóng góp</div>
          <div className="stat-value" style={{ color: "var(--accent)" }}>{loading ? "—" : submissions.length}</div>
          <div className="stat-sub">tin đã gửi</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Đang chờ duyệt</div>
          <div className="stat-value" style={{ color: "var(--warning)" }}>{loading ? "—" : pending}</div>
          <div className="stat-sub">pending</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Đã duyệt</div>
          <div className="stat-value" style={{ color: "var(--success)" }}>{loading ? "—" : approved}</div>
          <div className="stat-sub">được chấp nhận</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Bị từ chối</div>
          <div className="stat-value" style={{ color: "var(--danger)" }}>{loading ? "—" : rejected}</div>
          <div className="stat-sub">không hợp lệ</div>
        </div>
      </div>

      {/* Recent submissions */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="card-title" style={{ marginBottom: 0 }}>📋 Đóng góp gần đây</h2>
          <Link href="/dashboard/history" className="btn btn-ghost btn-sm">Xem tất cả →</Link>
        </div>
        {loading ? (
          <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)" }}>
            <div className="spinner" style={{ margin: "0 auto 0.5rem" }} />
            Đang tải...
          </div>
        ) : submissions.length === 0 ? (
          <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)" }}>
            <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📭</div>
            Bạn chưa đóng góp tin nào.<br />
            <Link href="/dashboard/submit" style={{ color: "var(--accent)", textDecoration: "none" }}>Đóng góp ngay →</Link>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tiêu đề</th>
                  <th>Nhãn</th>
                  <th>Trạng thái</th>
                  <th>Ngày gửi</th>
                </tr>
              </thead>
              <tbody>
                {submissions.slice(0, 5).map((s) => (
                  <tr key={s.id}>
                    <td style={{ maxWidth: "280px" }} className="truncate">{s.title}</td>
                    <td><span className={`badge badge-${s.user_label.toLowerCase()}`}>{s.user_label}</span></td>
                    <td><span className={`badge badge-${s.status}`}>{s.status}</span></td>
                    <td className="text-muted">{new Date(s.created_at).toLocaleDateString("vi-VN")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* About box */}
      <div className="card" style={{ marginTop: "1rem", background: "rgba(30,58,138,0.08)", border: "1px solid rgba(99,179,237,0.12)" }}>
        <p className="section-header">ℹ️ Về hệ thống</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "1rem", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
          <div>🤖 <strong>Mô hình:</strong> PhoBERT-base fine-tuned trên ViFN Dataset</div>
          <div>📚 <strong>Corpus:</strong> BKAI NewsCategory (596K bài VnExpress)</div>
          <div>🌐 <strong>Web RAG:</strong> DuckDuckGo → Báo Việt uy tín</div>
        </div>
      </div>
    </div>
  );
}
