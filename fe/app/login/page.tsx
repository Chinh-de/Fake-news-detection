"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [form, setForm] = useState({ username: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await authApi.login(form.username, form.password);
      localStorage.setItem("vifn_token", data.access_token);
      localStorage.setItem("vifn_user", JSON.stringify(data.user));
      router.push(data.user.role === "admin" ? "/admin" : "/dashboard");
    } catch (err: any) {
      setError(err.response?.data?.detail || "Đăng nhập thất bại. Vui lòng thử lại.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>🔍</div>
          <h1 className="logo-text" style={{ fontSize: "1.4rem", display: "block" }}>ViFN Platform</h1>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.3rem" }}>
            Vietnamese Fake News Detection
          </p>
        </div>

        <h2 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "1.5rem", textAlign: "center" }}>
          Đăng nhập
        </h2>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Tên đăng nhập</label>
            <input
              id="login-username"
              className="form-input"
              type="text"
              placeholder="Nhập username..."
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Mật khẩu</label>
            <input
              id="login-password"
              className="form-input"
              type="password"
              placeholder="Nhập mật khẩu..."
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
            />
          </div>
          <button
            id="login-submit"
            type="submit"
            className="btn btn-primary btn-full btn-lg"
            disabled={loading}
            style={{ marginTop: "0.5rem" }}
          >
            {loading ? <><span className="spinner" /> Đang đăng nhập...</> : "🔓 Đăng nhập"}
          </button>
        </form>

        <p style={{ textAlign: "center", marginTop: "1.5rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
          Chưa có tài khoản?{" "}
          <Link href="/register" style={{ color: "var(--accent)", textDecoration: "none", fontWeight: 600 }}>
            Đăng ký ngay
          </Link>
        </p>

        {/* Demo accounts */}
        <div style={{
          marginTop: "1.5rem",
          padding: "1rem",
          background: "rgba(59,130,246,0.06)",
          border: "1px solid rgba(59,130,246,0.15)",
          borderRadius: "var(--radius-sm)",
        }}>
          <p style={{ fontSize: "0.73rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
            Tài khoản demo
          </p>
          <div style={{ display: "flex", gap: "1rem", fontSize: "0.82rem" }}>
            <div>
              <span style={{ color: "var(--accent2)" }}>👑 Admin</span>
              <br /><code style={{ color: "var(--text-secondary)" }}>admin / admin123</code>
            </div>
            <div>
              <span style={{ color: "var(--success)" }}>👤 User</span>
              <br /><code style={{ color: "var(--text-secondary)" }}>demo / demo123</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
