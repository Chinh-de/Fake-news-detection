"use client";
import { useEffect, useState } from "react";
import { adminApi } from "@/lib/api";

export default function AdminUsersPage() {
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState<number | null>(null);
  const [modal, setModal] = useState<any>(null);

  const loadUsers = () => {
    setLoading(true);
    adminApi.users().then(setUsers).finally(() => setLoading(false));
  };
  useEffect(loadUsers, []);

  const handleUpdate = async (id: number, updates: { role?: string; is_active?: boolean }) => {
    setUpdating(id);
    try {
      const updated = await adminApi.updateUser(id, updates);
      setUsers(prev => prev.map(u => u.id === id ? updated : u));
    } catch (err: any) {
      alert(err.response?.data?.detail || "Cập nhật thất bại");
    } finally {
      setUpdating(null);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">👥 Quản lý Users</h1>
        <p className="page-sub">Xem, khóa tài khoản và thay đổi quyền của người dùng</p>
      </div>

      {/* Stats bar */}
      <div className="stat-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: "1.5rem" }}>
        <div className="stat-card">
          <div className="stat-label">Tổng users</div>
          <div className="stat-value" style={{ color: "var(--accent)" }}>{users.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Đang hoạt động</div>
          <div className="stat-value" style={{ color: "var(--success)" }}>{users.filter(u => u.is_active).length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Bị khóa</div>
          <div className="stat-value" style={{ color: "var(--danger)" }}>{users.filter(u => !u.is_active).length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Admin</div>
          <div className="stat-value" style={{ color: "var(--accent2)" }}>{users.filter(u => u.role === "admin").length}</div>
        </div>
      </div>

      <div className="card">
        {loading ? (
          <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-muted)" }}>
            <div className="spinner" style={{ margin: "0 auto 0.5rem", width: "32px", height: "32px" }} />
            Đang tải...
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Tài khoản</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Trạng thái</th>
                  <th>Ngày tạo</th>
                  <th>Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id}>
                    <td className="text-muted">#{u.id}</td>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{
                          width: "28px", height: "28px", borderRadius: "50%",
                          background: u.role === "admin" ? "linear-gradient(135deg, #8b5cf6, #3b82f6)" : "var(--bg-glass)",
                          border: "1px solid var(--border)",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: "0.7rem", fontWeight: 800, color: "#fff",
                        }}>
                          {u.username[0].toUpperCase()}
                        </div>
                        <span style={{ fontWeight: 500, color: "var(--text-primary)" }}>{u.username}</span>
                      </div>
                    </td>
                    <td className="text-secondary text-sm">{u.email}</td>
                    <td>
                      <span className={`badge badge-${u.role}`}>
                        {u.role === "admin" ? "👑 Admin" : "👤 User"}
                      </span>
                    </td>
                    <td>
                      <span className={u.is_active ? "badge badge-approved" : "badge badge-rejected"}>
                        {u.is_active ? "✅ Active" : "🔒 Locked"}
                      </span>
                    </td>
                    <td className="text-muted text-sm">{new Date(u.created_at).toLocaleDateString("vi-VN")}</td>
                    <td>
                      <div style={{ display: "flex", gap: "0.4rem" }}>
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => setModal(u)}
                          title="Xem & chỉnh sửa"
                        >
                          ✏️
                        </button>
                        <button
                          className={`btn btn-sm ${u.is_active ? "btn-danger" : "btn-success"}`}
                          onClick={() => handleUpdate(u.id, { is_active: !u.is_active })}
                          disabled={updating === u.id}
                          title={u.is_active ? "Khóa tài khoản" : "Mở khóa tài khoản"}
                        >
                          {updating === u.id ? <span className="spinner" style={{ width: "12px", height: "12px" }} /> :
                            u.is_active ? "🔒" : "🔓"}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Edit modal */}
      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="modal-title">✏️ Chỉnh sửa User #{modal.id}</h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setModal(null)}>✕</button>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <div className="form-label">Tài khoản</div>
              <div style={{ fontWeight: 600 }}>{modal.username} · <span style={{ color: "var(--text-muted)" }}>{modal.email}</span></div>
            </div>

            <div className="form-group">
              <label className="form-label">Role</label>
              <select className="form-select" defaultValue={modal.role}
                onChange={e => { setModal({ ...modal, role: e.target.value }); }}>
                <option value="user">👤 User</option>
                <option value="admin">👑 Admin</option>
              </select>
            </div>

            <div style={{ display: "flex", gap: "0.75rem", marginTop: "1.5rem" }}>
              <button className="btn btn-primary" style={{ flex: 1 }}
                disabled={updating === modal.id}
                onClick={async () => {
                  await handleUpdate(modal.id, { role: modal.role });
                  setModal(null);
                }}>
                {updating === modal.id ? <><span className="spinner" /> Đang lưu...</> : "💾 Lưu thay đổi"}
              </button>
              <button className="btn btn-ghost" onClick={() => setModal(null)}>Hủy</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
