import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: {
    outDir: fileURLToPath(new URL("../api/luckymodels_api/ui/dist", import.meta.url)),
    emptyOutDir: true,
  },
});
