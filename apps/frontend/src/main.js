import { createApp } from 'vue'
import './assets/global.css'
import App from './App.vue'
import keycloak, { initKeycloak, startTokenRefresh } from './auth/keycloak'
import router from './router/router'

const bootstrap = async () => {
  await initKeycloak()
  startTokenRefresh()

  const app = createApp(App)

  app.provide('keycloak', keycloak)
  app.use(router)

  app.mount('#genPM')
}

bootstrap().catch((error) => {
  console.error('Authentication failed', error)

  const host = document.getElementById('genPM')
  if (host) {
    host.innerHTML = '<p>Authentication failed</p>'
  }
})
