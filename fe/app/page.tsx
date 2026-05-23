"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    const token = localStorage.getItem("vifn_token");
    const user = localStorage.getItem("vifn_user");
    if (!token) {
      router.replace("/login");
    } else {
      try {
        const u = JSON.parse(user || "{}");
        router.replace(u.role === "admin" ? "/admin" : "/dashboard");
      } catch {
        router.replace("/dashboard");
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div className="spinner" />
    </div>
  );
}
