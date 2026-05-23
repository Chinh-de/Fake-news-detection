"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/Sidebar";

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  useEffect(() => {
    const token = localStorage.getItem("vifn_token");
    const user = localStorage.getItem("vifn_user");
    if (!token) { router.replace("/login"); return; }
    try {
      const u = JSON.parse(user || "{}");
      if (u.role !== "admin") router.replace("/dashboard");
    } catch { router.replace("/login"); }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">{children}</main>
    </div>
  );
}
