import { computed, ref } from 'vue';
import Keycloak from 'keycloak-js';

const keycloakUrl = import.meta.env.VITE_KEYCLOAK_URL;
const keycloakRealm = import.meta.env.VITE_KEYCLOAK_REALM;
const keycloakClientId = import.meta.env.VITE_KEYCLOAK_CLIENT_ID;
const adminRole =
  (import.meta.env.VITE_KEYCLOAK_ADMIN_ROLE ?? 'admin').trim() || 'admin';

const missingVariables = [];

if (!keycloakUrl) missingVariables.push('VITE_KEYCLOAK_URL');
if (!keycloakRealm) missingVariables.push('VITE_KEYCLOAK_REALM');
if (!keycloakClientId) missingVariables.push('VITE_KEYCLOAK_CLIENT_ID');

const keycloak = new Keycloak({
  url: keycloakUrl,
  realm: keycloakRealm,
  clientId: keycloakClientId,
});

let refreshIntervalId = null;
let authFailureHandled = false;

export const authRoles = ref(new Set());

const getTokenRoles = () => {
  const parsed = keycloak.tokenParsed;
  if (!parsed) return new Set();

  const realmRoles = parsed.realm_access?.roles ?? [];
  const clientRoles = parsed.resource_access?.[keycloakClientId]?.roles ?? [];
  return new Set([...realmRoles, ...clientRoles]);
};

const syncAuthRoles = () => {
  authRoles.value = getTokenRoles();
};

export const isAdmin = computed(() => authRoles.value.has(adminRole));

export const hasAdminRole = () => isAdmin.value;

const refreshToken = async () => {
  try {
    await keycloak.updateToken(30);
    syncAuthRoles();
    authFailureHandled = false;
  } catch (_error) {
    if (!authFailureHandled) {
      authFailureHandled = true;
      stopTokenRefresh();
      keycloak.clearToken();
      syncAuthRoles();
    }
    throw new Error('Session expired. Please sign in again.');
  }
};

export const initKeycloak = async () => {
  const authenticated = await keycloak.init({
    onLoad: 'login-required',
    checkLoginIframe: false,
    pkceMethod: 'S256',
  });

  if (missingVariables.length > 0) {
    throw new Error(`Missing required Keycloak variables: ${missingVariables.join(', ')}`);
  }

  if (!authenticated) {
    throw new Error('User is not authenticated');
  }

  syncAuthRoles();
};

export const startTokenRefresh = () => {
  if (refreshIntervalId !== null) {
    return;
  }

  refreshIntervalId = window.setInterval(() => {
    void refreshToken();
  }, 20000);
};

export const stopTokenRefresh = () => {
  if (refreshIntervalId === null) {
    return;
  }

  window.clearInterval(refreshIntervalId);
  refreshIntervalId = null;
};

export const getAccessToken = async () => {
  await refreshToken();

  if (!keycloak.token) {
    throw new Error('Missing access token');
  }

  return keycloak.token;
};

export const getAuthProfile = () => ({
  username: keycloak.tokenParsed?.preferred_username ?? '',
  fullName: keycloak.tokenParsed?.name ?? '',
});

export const logout = async () => {
  stopTokenRefresh();
  authRoles.value = new Set();
  await keycloak.logout({ redirectUri: window.location.origin });
};

export default keycloak;
