import Keycloak from 'keycloak-js';

const keycloakUrl = import.meta.env.VITE_KEYCLOAK_URL;
const keycloakRealm = import.meta.env.VITE_KEYCLOAK_REALM;
const keycloakClientId = import.meta.env.VITE_KEYCLOAK_CLIENT_ID;

const missingVariables = [];

if (!keycloakUrl) missingVariables.push('VITE_KEYCLOAK_URL');
if (!keycloakRealm) missingVariables.push('VITE_KEYCLOAK_REALM');
if (!keycloakClientId) missingVariables.push('VITE_KEYCLOAK_CLIENT_ID');

if (missingVariables.length > 0) {
  throw new Error(`Missing required Keycloak variables: ${missingVariables.join(', ')}`);
}

const keycloak = new Keycloak({
  url: keycloakUrl,
  realm: keycloakRealm,
  clientId: keycloakClientId,
});

let refreshIntervalId = null;

const refreshToken = async () => {
  try {
    await keycloak.updateToken(30);
  } catch (_error) {
    await keycloak.login();
  }
};

export const initKeycloak = async () => {
  const authenticated = await keycloak.init({
    onLoad: 'login-required',
    checkLoginIframe: false,
    pkceMethod: 'S256',
  });

  if (!authenticated) {
    throw new Error('User is not authenticated');
  }
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
  await keycloak.logout({ redirectUri: window.location.origin });
};

export default keycloak;
