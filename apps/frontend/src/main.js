import { createApp } from 'vue'
import './assets/global.css'
import App from './App.vue'
import keycloak, { initKeycloak, startTokenRefresh } from './auth/keycloak'

const bootstrap = async () => {
  await initKeycloak()
  startTokenRefresh()

  createApp(App)
    .provide('keycloak', keycloak)
    .mount('#genPM')
}

bootstrap().catch((error) => {
  console.error('Authentication failed', error)

  const host = document.getElementById('genPM')
  if (host) {
    host.innerHTML = '<p>Authentication failed</p>'
  }
})
