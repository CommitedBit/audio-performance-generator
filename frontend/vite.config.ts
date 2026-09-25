import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Same-origin /api in dev as in the nginx image, so VITE_API_BASE can
      // stay '/api' everywhere and CORS never comes up.
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
        // Music generation runs for minutes.
        timeout: 600_000,
        proxyTimeout: 600_000,
      },
    },
  },
})
