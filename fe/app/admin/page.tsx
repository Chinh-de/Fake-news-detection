"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { adminApi } from "@/lib/api";

export default function AdminDashboard() {
  const [stats, setStats] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([adminApi.stats(), adminApi.retrainJobs()])
      .then(([s, j]) => { setStats(s); setJobs(j.slice(0, 5)); })
      .finally(() => setLoading(false));
  }, []);

  const statCards = stats ? [
    { label: "Tổng Users", value: stats.total_users, color: "var(--accent)", icon: "👥", link: "/admin/users" },
    { label: "Tổng Submissions", value: stats.total_submissions, color: "var(--accent2)", icon: "📥", link: "/admin/submissions" },
    { label: "Chờ duyệt", value: stats.pending_count, color: "var(--warning)", icon: "⏳", link: "/admin/submissions?status=pending" },
    { label: "Đã duyệt", value: stats.approved_count, color: "var(--success)", icon: "✅", link: "/admin/submissions?status=approved" },
    { label: "Từ chối", value: stats.rejected_count, color: "var(--danger)", icon: "❌", link: "/admin/submissions?status=rejected" },
    { label: "Retrain Jobs", value: stats.total_retrain_jobs, color: "#e879f9", icon: "🤖", link: "/admin/retrain" },
  ] : [];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">📊 Admin Dashboard</h1>
        <p className="page-sub">Quản lý hệ thống ViFN Fake News Detection Platform</p>
      </div>

      {/* Hero banner */}
      <div style={{
        background: "linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f3460 100%)",
        border: "1px solid rgba(99,179,237,0.15)",
        borderRadius: "var(--radius-lg)",
        padding: "1.5rem 2rem",
        marginBottom: "2rem",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <div style={{
            fontSize: "1.4rem", fontWeight: 800,
            background: "linear-gradient(90deg, #63b3ed, #90cdf4, #bee3f8)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            🔍 ViFN Fake News Platform
          </div>
          <div style={{ color: "#94a3b8", fontSize: "0.88rem", marginTop: "0.3rem" }}>
            PhoBERT-base · BM25 Retrieval · Web RAG · SQLite · FastAPI
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <Link href="/admin/submissions?status=pending" className="btn btn-primary">
            ⏳ Duyệt tin ({stats?.pending_count ?? "..."})
          </Link>
          <Link href="/admin/retrain" className="btn btn-ghost">🤖 Retrain Model</Link>
        </div>
      </div>

      {/* Stat cards */}
      {loading ? (
        <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>
          <div className="spinner" style={{ margin: "0 auto 0.5rem", width: "32px", height: "32px" }} />
          Đang tải thống kê...
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "2rem" }}>
          {statCards.map((s, i) => (
            <Link key={i} href={s.link} style={{ textDecoration: "none" }}>
              <div className="stat-card" style={{ cursor: "pointer" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div className="stat-label">{s.label}</div>
                  <div style={{ fontSize: "1.3rem" }}>{s.icon}</div>
                </div>
                <div className="stat-value" style={{ color: s.color }}>{s.value}</div>
                <div className="stat-sub">Bấm để xem chi tiết →</div>
              </div>
            </Link>
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Recent retrain jobs */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="card-title" style={{ marginBottom: 0 }}>🤖 Retrain gần đây</h2>
            <Link href="/admin/retrain" className="btn btn-ghost btn-sm">Xem tất cả →</Link>
          </div>
          {jobs.length === 0 ? (
            <div style={{ textAlign: "center", padding: "1.5rem", color: "var(--text-muted)" }}>
              Chưa có retrain job nào
            </div>
          ) : (
            <div>
              {jobs.map(j => (
                <div key={j.id} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "0.65rem 0", borderBottom: "1px solid var(--border)",
                }}>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: "0.88rem" }}>Job #{j.id}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      {new Date(j.created_at).toLocaleString("vi-VN")}
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <span className={`badge badge-${j.status}`}>{j.status}</span>
                    {j.accuracy_before && j.accuracy_after && (
                      <div style={{ fontSize: "0.73rem", color: "var(--success)", marginTop: "0.2rem" }}>
                        {(j.accuracy_before * 100).toFixed(1)}% → {(j.accuracy_after * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quick links */}
        <div className="card">
          <h2 className="card-title">⚡ Thao tác nhanh</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <Link href="/admin/submissions?status=pending" className="btn btn-primary" style={{ justifyContent: "flex-start" }}>
              📥 Xem submission chờ duyệt ({stats?.pending_count ?? "..."})
            </Link>
            <Link href="/admin/users" className="btn btn-ghost" style={{ justifyContent: "flex-start" }}>
              👥 Quản lý tài khoản users
            </Link>
            <Link href="/admin/retrain" className="btn btn-ghost" style={{ justifyContent: "flex-start" }}>
              🤖 Khởi động retrain model
            </Link>
            <Link href="/dashboard/predict" className="btn btn-ghost" style={{ justifyContent: "flex-start" }}>
              🔍 Thử phân tích tin tức
            </Link>
          </div>

          <div style={{ marginTop: "1.5rem" }}>
            <div className="section-header">📈 Retrain Progress</div>
            {stats?.last_retrain_at ? (
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                Lần retrain cuối: <strong>{new Date(stats.last_retrain_at).toLocaleString("vi-VN")}</strong>
              </p>
            ) : (
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Chưa có lần retrain nào</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
