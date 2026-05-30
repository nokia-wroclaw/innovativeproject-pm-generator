import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse, urlunparse

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError


@dataclass(frozen=True)
class KeycloakSettings:
    server_url: str
    issuer_url: str
    realm: str
    client_id: str
    required_roles: tuple[str, ...]
    admin_role: str

    @property
    def issuer(self) -> str:
        return f"{self.issuer_url}/realms/{self.realm}"

    @property
    def jwks_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/certs"


def _required_env(name: str) -> str:
    if value := os.getenv(name):
        return value

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Missing required Keycloak configuration: {name}",
    )


@lru_cache
def get_keycloak_settings() -> KeycloakSettings:
    server_url = _required_env("KEYCLOAK_SERVER_URL").rstrip("/")
    issuer_url = os.getenv("KEYCLOAK_ISSUER_URL", server_url).rstrip("/")
    realm = _required_env("KEYCLOAK_REALM")
    client_id = _required_env("KEYCLOAK_CLIENT_ID")
    required_roles_raw = os.getenv("KEYCLOAK_REQUIRED_ROLES", "")
    required_roles = tuple(role.strip() for role in required_roles_raw.split(",") if role.strip())
    admin_role = os.getenv("KEYCLOAK_ADMIN_ROLE", "admin").strip() or "admin"
    return KeycloakSettings(
        server_url=server_url,
        issuer_url=issuer_url,
        realm=realm,
        client_id=client_id,
        required_roles=required_roles,
        admin_role=admin_role,
    )


@lru_cache
def get_jwk_client() -> PyJWKClient:
    return PyJWKClient(get_keycloak_settings().jwks_url)


bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def _extract_roles(payload: dict[str, Any], client_id: str) -> set[str]:
    realm_roles = payload.get("realm_access", {}).get("roles", [])
    client_roles = payload.get("resource_access", {}).get(client_id, {}).get("roles", [])
    return set(realm_roles) | set(client_roles)


def _with_host_variant(base_url: str, host: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return base_url

    hostname = parsed.hostname or ""
    if not hostname:
        return base_url

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _candidate_issuer_roots(settings: KeycloakSettings) -> set[str]:
    roots = {settings.issuer_url.rstrip("/"), settings.server_url.rstrip("/")}
    expanded: set[str] = set()
    for root in roots:
        expanded.add(root)
        expanded.add(_with_host_variant(root, "localhost"))
        expanded.add(_with_host_variant(root, "127.0.0.1"))
    return {root.rstrip("/") for root in expanded if root}


def _allowed_issuers(settings: KeycloakSettings) -> set[str]:
    return {f"{root}/realms/{settings.realm}" for root in _candidate_issuer_roots(settings)}


def get_token_payload(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if not credentials:
        raise _unauthorized("Missing bearer token")

    token = credentials.credentials
    settings = get_keycloak_settings()

    try:
        signing_key = get_jwk_client().get_signing_key_from_jwt(token).key
    except PyJWKClientError as exc:
        raise _unauthorized(
            f"Invalid bearer token (JWKS/signature key lookup failed at {settings.jwks_url})"
        ) from exc

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            options={
                "require": ["exp", "iat"],
                "verify_aud": False,
                "verify_iss": False,
            },
            leeway=60,
        )
    except InvalidTokenError as exc:
        raise _unauthorized(f"Invalid bearer token ({exc})") from exc

    token_issuer = str(payload.get("iss", "")).rstrip("/")
    if token_issuer not in _allowed_issuers(settings):
        raise _unauthorized("Invalid token issuer")

    if str(payload.get("typ", "Bearer")).lower() != "bearer":
        raise _unauthorized("Invalid token type")

    token_aud = payload.get("aud")
    aud_matches = token_aud == settings.client_id or (
            isinstance(token_aud, list) and settings.client_id in token_aud
    )

    if (authorized_party := str(payload.get("azp", ""))) and authorized_party != settings.client_id:
        raise _unauthorized("Invalid token client")

    if not aud_matches and authorized_party != settings.client_id:
        raise _unauthorized("Invalid token audience/client")

    user_id = payload.get("sub") or payload.get("clientId") or payload.get("client_id")

    if not user_id:
        print("WARNING: Missing user identification claims. Payload:", payload)
        raise _unauthorized("Token payload is missing identification claims (sub/clientId)")

    session_id = payload.get("sid") or payload.get("session_state")

    payload["user_id"] = str(user_id)

    payload["session_id"] = str(session_id) if session_id else None

    return payload


def require_auth(payload: dict[str, Any] = Depends(get_token_payload)) -> dict[str, Any]:
    settings = get_keycloak_settings()
    if settings.required_roles:
        token_roles = _extract_roles(payload, settings.client_id)
        missing_roles = [role for role in settings.required_roles if role not in token_roles]
        if missing_roles:
            raise _forbidden("Insufficient permissions")
    return payload


def require_admin(payload: dict[str, Any] = Depends(require_auth)) -> dict[str, Any]:
    settings = get_keycloak_settings()
    token_roles = _extract_roles(payload, settings.client_id)
    if settings.admin_role not in token_roles:
        raise _forbidden("Admin role required to delete datasets")
    return payload
