import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ViFN — Vietnamese Fake News Detection Platform",
  description:
    "Nền tảng phát hiện tin giả tiếng Việt sử dụng PhoBERT + Retrieval-Augmented Generation. Đăng ký, phân tích tin và đóng góp cho mô hình.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
