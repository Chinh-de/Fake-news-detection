"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface NavItem {
  href: string;
  icon: string;
  label: string;
}

const userNav: NavItem[] = [
  { href: "/dashboard", icon: "🏠", label: "Tổng quan" },
  { href: "/dashboard/predict", icon: "🔍", label: "Phân tích tin" },
  { href: "/dashboard/submit", icon: "📤", label: "Đóng góp tin" },
  { href: "/dashboard/history", icon: "📋", label: "Lịch sử" },
];

const adminNav: NavItem[] = [
  { href: "/admin", icon: "📊", label: "Dashboard" },
  { href: "/admin/submissions", icon: "📥", label: "Duyệt tin" },
  { href: "/admin/users", icon: "👥", label: "Quản lý Users" },
  { href: "/admin/retrain", icon: "🤖", label: "Retrain Model" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    const u = localStorage.getItem("vifn_user");
    if (u) setUser(JSON.parse(u));
  }, []);

  const isAdmin = user?.role === "admin";
  const navItems = isAdmin ? adminNav : userNav;

  const handleLogout = () => {
    localStorage.removeItem("vifn_token");
    localStorage.removeItem("vifn_user");
    router.push("/login");
  };

  return (
    <div className="sidebar">
      {/* Logo */}
      <div className="logo">
        <span className="logo-icon">🔍</span>
        <span className="logo-text">ViFN</span>
      </div>

      {/* User chip */}
      {user && (
        <div style={{ padding: "0 1rem", marginBottom: "1rem" }}>
          <div className="user-chip">
            <div className="user-avatar">
              {user.username?.[0]?.toUpperCase()}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: "0.82rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {user.username}
              </div>
              <div>
                <span className={`badge badge-${user.role} text-xs`} style={{ fontSize: "0.65rem" }}>
                  {user.role === "admin" ? "👑 Admin" : "👤 User"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Nav */}
      <div className="nav-section" style={{ flex: 1 }}>
        <div className="nav-section-label">
          {isAdmin ? "Quản trị" : "Người dùng"}
        </div>
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-item ${pathname === item.href ? "active" : ""}`}
          >
            <span className="nav-item-icon">{item.icon}</span>
            {item.label}
          </Link>
        ))}

        {/* Switch view for admin */}
        {isAdmin && (
          <>
            <div className="nav-section-label" style={{ marginTop: "1rem" }}>Chế độ xem</div>
            <Link href="/dashboard" className="nav-item">
              <span className="nav-item-icon">👤</span>
              Giao diện User
            </Link>
          </>
        )}
        {!isAdmin && (
          <>
            <div className="nav-section-label" style={{ marginTop: "1rem" }}>Tài khoản</div>
          </>
        )}
      </div>

      {/* Logout */}
      <div style={{ padding: "0 1rem 1rem" }}>
        <button
          onClick={handleLogout}
          className="nav-item btn-ghost"
          style={{ width: "100%", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)" }}
        >
          <span className="nav-item-icon">🚪</span>
          Đăng xuất
        </button>
      </div>
    </div>
  );
}
