import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'

// The UI's version, built in from package.json so the Settings page can show
// it beside the server's -- a mismatched pair is the first thing to check.
const { version } = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8'))

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(version),
  },
  server: {
    proxy: {
      // Same-origin /api in dev as in the nginx image, so VITE_API_BASE can
      // stay '/api' everywhere and CORS never comes up.
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
        // Longer than the gateway's FORWARD_TIMEOUT (600 s), matching nginx.
        timeout: 660_000,
        proxyTimeout: 660_000,
      },
    },
  },
})
