"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const [form, setForm] = useState({ username: "", email: "", password: "", confirm: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (form.password !== form.confirm) {
      setError("Mật khẩu xác nhận không khớp");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await authApi.register(form.username, form.email, form.password);
      setSuccess(true);
      setTimeout(() => router.push("/login"), 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Đăng ký thất bại. Vui lòng thử lại.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>🔍</div>
          <h1 className="logo-text" style={{ fontSize: "1.4rem", display: "block" }}>ViFN Platform</h1>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.3rem" }}>
            Vietnamese Fake News Detection
          </p>
        </div>

        <h2 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "1.5rem", textAlign: "center" }}>
          Tạo tài khoản mới
        </h2>

        {error && <div className="alert alert-error">{error}</div>}
        {success && (
          <div className="alert alert-success">
            ✅ Đăng ký thành công! Đang chuyển hướng đến trang đăng nhập...
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Tên đăng nhập</label>
            <input
              id="reg-username"
              className="form-input"
              type="text"
              placeholder="Ít nhất 3 ký tự..."
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              required minLength={3}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Email</label>
            <input
              id="reg-email"
              className="form-input"
              type="email"
              placeholder="example@email.com"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Mật khẩu</label>
            <input
              id="reg-password"
              className="form-input"
              type="password"
              placeholder="Ít nhất 6 ký tự..."
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required minLength={6}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Xác nhận mật khẩu</label>
            <input
              id="reg-confirm"
              className="form-input"
              type="password"
              placeholder="Nhập lại mật khẩu..."
              value={form.confirm}
              onChange={(e) => setForm({ ...form, confirm: e.target.value })}
              required
            />
          </div>
          <button
            id="reg-submit"
            type="submit"
            className="btn btn-primary btn-full btn-lg"
            disabled={loading || success}
            style={{ marginTop: "0.5rem" }}
          >
            {loading ? <><span className="spinner" /> Đang tạo tài khoản...</> : "✨ Đăng ký"}
          </button>
        </form>

        <p style={{ textAlign: "center", marginTop: "1.5rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
          Đã có tài khoản?{" "}
          <Link href="/login" style={{ color: "var(--accent)", textDecoration: "none", fontWeight: 600 }}>
            Đăng nhập
          </Link>
        </p>
      </div>
    </div>
  );
}
