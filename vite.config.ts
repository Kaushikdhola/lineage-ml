import { defineConfig } from "vitest/config";

export default defineConfig({
  build: {
    target: "es2022",
    sourcemap: true,
    // three.js alone is ~600 kB minified; split vendors so app code stays small and cacheable
    chunkSizeWarningLimit: 700,
    rollupOptions: { output: { manualChunks: { three: ["three"], katex: ["katex"] } } },
  },
  test: {
    include: ["tests/**/*.test.ts"],
    testTimeout: 30_000,
  },
});
