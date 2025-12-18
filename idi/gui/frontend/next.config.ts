import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  turbopack: {
    // This repo has multiple lockfiles; ensure Turbopack treats the GUI
    // frontend directory as the workspace root.
    root: process.cwd(),
  },
};

export default nextConfig;
