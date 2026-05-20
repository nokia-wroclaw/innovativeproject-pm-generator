import { createApp } from 'vue'
import './assets/global.css'
import './assets/tailwind.css'
import { VueQueryPlugin, QueryClient } from '@tanstack/vue-query'
import App from './App.vue'
import keycloak, { initKeycloak, startTokenRefresh } from './auth/keycloak'
import router from './router/router'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      refetchOnWindowFocus: true,
      retry: 1,
    },
    mutations: {
      retry: 0,
    },
  },
})

const bootstrap = async () => {
  await initKeycloak()
  startTokenRefresh()

  const app = createApp(App)

  app.provide('keycloak', keycloak)
  app.use(router)
  app.use(VueQueryPlugin, { queryClient })

  app.mount('#genPM')
}

bootstrap().catch((error) => {
  console.error('Authentication failed', error)

  const host = document.getElementById('genPM')
  if (host) {
    host.innerHTML = '<p>Authentication failed</p>'
  }
})
