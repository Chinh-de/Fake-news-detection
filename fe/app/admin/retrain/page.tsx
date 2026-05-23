"use client";
import { useEffect, useState, useCallback } from "react";
import { adminApi } from "@/lib/api";

export default function AdminRetrainPage() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [approvedSubs, setApprovedSubs] = useState<any[]>([]);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [pollingJobId, setPollingJobId] = useState<number | null>(null);
  const [logRefresh, setLogRefresh] = useState(0);

  const loadData = useCallback(() => {
    Promise.all([
      adminApi.retrainJobs(),
      adminApi.submissions("approved"),
    ]).then(([j, s]) => {
      setJobs(j);
      setApprovedSubs(s);
      // Auto-poll if there's a running/queued job
      const running = j.find((jb: any) => jb.status === "queued" || jb.status === "running");
      if (running) setPollingJobId(running.id);
      else setPollingJobId(null);
    }).finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Poll running job every 3s
  useEffect(() => {
    if (!pollingJobId) return;
    const interval = setInterval(async () => {
      try {
        const job = await adminApi.getRetrainJob(pollingJobId);
        setJobs(prev => prev.map(j => j.id === pollingJobId ? job : j));
        if (activeJob?.id === pollingJobId) setActiveJob(job);
        if (job.status !== "queued" && job.status !== "running") {
          setPollingJobId(null);
        }
        setLogRefresh(r => r + 1);
      } catch {}
    }, 3000);
    return () => clearInterval(interval);
  }, [pollingJobId, activeJob]);

  const handleStartRetrain = async () => {
    if (selectedIds.length === 0) { alert("Chọn ít nhất 1 submission đã được approve"); return; }
    if (!confirm(`Bắt đầu retrain với ${selectedIds.length} submission đã chọn?`)) return;
    setStarting(true);
    try {
      const job = await adminApi.startRetrain(selectedIds);
      setJobs(prev => [job, ...prev]);
      setSelectedIds([]);
      setActiveJob(job);
      setPollingJobId(job.id);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Không thể bắt đầu retrain");
    } finally {
      setStarting(false);
    }
  };

  const toggleSelect = (id: number) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🤖 Retrain Model</h1>
        <p className="page-sub">Chọn các submission đã duyệt và khởi động fine-tuning PhoBERT</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Left: Select submissions */}
        <div>
          <div className="card">
            <div className="section-header">📋 Submissions đã approve ({approvedSubs.length})</div>
            <p style={{ fontSize: "0.83rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
              Chọn tin tức muốn đưa vào tập huấn luyện để retrain model
            </p>

            {approvedSubs.length === 0 ? (
              <div style={{
                textAlign: "center", padding: "2rem",
                background: "var(--bg-glass)", borderRadius: "var(--radius-sm)",
                color: "var(--text-muted)", fontSize: "0.85rem",
              }}>
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📭</div>
                Không có submission nào đã được approve.<br />
                <a href="/admin/submissions" style={{ color: "var(--accent)", textDecoration: "none" }}>
                  Đi duyệt submissions →
                </a>
              </div>
            ) : (
              <div style={{ maxHeight: "360px", overflowY: "auto", marginBottom: "1rem" }}>
                {approvedSubs.map(s => (
                  <label key={s.id} style={{
                    display: "flex", gap: "0.75rem",
                    padding: "0.65rem 0.5rem",
                    borderBottom: "1px solid var(--border)",
                    cursor: "pointer",
                    background: selectedIds.includes(s.id) ? "rgba(59,130,246,0.06)" : "transparent",
                    borderRadius: selectedIds.includes(s.id) ? "var(--radius-sm)" : undefined,
                    transition: "background 0.15s",
                  }}>
                    <input type="checkbox" checked={selectedIds.includes(s.id)}
                      onChange={() => toggleSelect(s.id)}
                      style={{ accentColor: "var(--accent)", marginTop: "2px" }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="truncate" style={{ fontWeight: 500, fontSize: "0.87rem", color: "var(--text-primary)" }}>
                        {s.title}
                      </div>
                      <div style={{ fontSize: "0.73rem", color: "var(--text-muted)", display: "flex", gap: "0.5rem", marginTop: "0.2rem" }}>
                        <span className={`badge badge-${s.user_label.toLowerCase()}`} style={{ fontSize: "0.65rem" }}>{s.user_label}</span>
                        <span>{s.author?.username}</span>
                        <span>{new Date(s.created_at).toLocaleDateString("vi-VN")}</span>
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            )}

            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
              <button className="btn btn-ghost btn-sm"
                onClick={() => setSelectedIds(approvedSubs.map(s => s.id))}>
                ✅ Chọn tất cả
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => setSelectedIds([])}>
                ☐ Bỏ chọn
              </button>
            </div>

            <button
              id="start-retrain"
              className="btn btn-primary btn-full"
              style={{ padding: "0.75rem" }}
              onClick={handleStartRetrain}
              disabled={starting || selectedIds.length === 0 || pollingJobId !== null}
            >
              {starting ? <><span className="spinner" /> Đang khởi động...</> :
                pollingJobId ? "⏳ Đang có job đang chạy..." :
                  `🚀 Bắt đầu Retrain (${selectedIds.length} submission)`}
            </button>

            {pollingJobId && (
              <div className="alert alert-info" style={{ marginTop: "0.75rem", fontSize: "0.82rem" }}>
                🔄 Đang retrain job #{pollingJobId} — tự động cập nhật sau mỗi 3 giây...
              </div>
            )}
          </div>
        </div>

        {/* Right: Jobs + Log */}
        <div>
          {/* Job detail / active log */}
          {activeJob && (
            <div className="card" style={{ marginBottom: "1rem" }}>
              <div className="flex items-center justify-between mb-2">
                <div className="section-header" style={{ marginBottom: 0 }}>
                  📊 Job #{activeJob.id} — <span className={`badge badge-${activeJob.status}`}>{activeJob.status}</span>
                </div>
                <button className="btn btn-ghost btn-sm" onClick={() => setActiveJob(null)}>✕</button>
              </div>

              {activeJob.accuracy_before && (
                <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.75rem" }}>
                  <div>
                    <div className="text-xs text-muted">Accuracy trước</div>
                    <div style={{ fontWeight: 700, color: "var(--warning)" }}>
                      {(activeJob.accuracy_before * 100).toFixed(2)}%
                    </div>
                  </div>
                  {activeJob.accuracy_after && (
                    <div>
                      <div className="text-xs text-muted">Accuracy sau</div>
                      <div style={{ fontWeight: 700, color: "var(--success)" }}>
                        {(activeJob.accuracy_after * 100).toFixed(2)}% ↑
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeJob.log_text && (
                <div className="log-box" key={logRefresh}>
                  {activeJob.log_text}
                </div>
              )}

              {(activeJob.status === "queued" || activeJob.status === "running") && (
                <div style={{ textAlign: "center", marginTop: "0.75rem" }}>
                  <div className="spinner" style={{ margin: "0 auto" }} />
                  <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.4rem" }}>
                    Đang chạy... tự động cập nhật
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Jobs history */}
          <div className="card">
            <div className="section-header">📜 Lịch sử Retrain Jobs</div>
            {loading ? (
              <div style={{ textAlign: "center", padding: "2rem" }}>
                <div className="spinner" style={{ margin: "0 auto" }} />
              </div>
            ) : jobs.length === 0 ? (
              <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Chưa có retrain job nào
              </div>
            ) : (
              <div>
                {jobs.map(j => (
                  <div key={j.id} style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "0.75rem 0", borderBottom: "1px solid var(--border)",
                    cursor: "pointer",
                  }}
                    onClick={() => setActiveJob(j)}>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: "0.88rem" }}>
                        Job #{j.id}
                        <span className={`badge badge-${j.status}`} style={{ marginLeft: "0.5rem" }}>{j.status}</span>
                      </div>
                      <div className="text-xs text-muted">
                        {new Date(j.created_at).toLocaleString("vi-VN")}
                        {j.triggered_by_user && ` · ${j.triggered_by_user.username}`}
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      {j.accuracy_before && (
                        <div style={{ fontSize: "0.78rem" }}>
                          <span style={{ color: "var(--warning)" }}>{(j.accuracy_before * 100).toFixed(1)}%</span>
                          {j.accuracy_after && (
                            <> → <span style={{ color: "var(--success)" }}>{(j.accuracy_after * 100).toFixed(1)}%</span></>
                          )}
                        </div>
                      )}
                      <div className="text-xs text-muted">
                        {JSON.parse(j.submission_ids || "[]").length} submissions
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
