import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "IDI/IAN Interface",
  description: "Intelligent Daemon Interface GUI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-background antialiased selection:bg-cyan-500/30`}>
        <Navbar />
        <main className="container mx-auto p-4 md:p-8">
          {children}
        </main>
      </body>
    </html>
  );
}
