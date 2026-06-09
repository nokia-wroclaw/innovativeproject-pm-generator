import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

const plotlyMin = fileURLToPath(
  new URL('./node_modules/plotly.js-dist-min/plotly.min.js', import.meta.url),
)

export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      'plotly.js-dist-min': plotlyMin,
    },
  },
  optimizeDeps: {
    include: ['plotly.js-dist-min'],
  },
})
