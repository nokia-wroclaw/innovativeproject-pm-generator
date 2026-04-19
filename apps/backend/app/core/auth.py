import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

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

    @property
    def issuer(self) -> str:
        return f"{self.issuer_url}/realms/{self.realm}"

    @property
    def jwks_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/certs"


def _required_env(name: str) -> str:
    if value := os.getenv(name):
        return value

    raise ValueError(f"{name} environment variable is required")


@lru_cache
def get_keycloak_settings() -> KeycloakSettings:
    server_url = _required_env("KEYCLOAK_SERVER_URL").rstrip("/")
    issuer_url = os.getenv("KEYCLOAK_ISSUER_URL", server_url).rstrip("/")
    realm = _required_env("KEYCLOAK_REALM")
    client_id = _required_env("KEYCLOAK_CLIENT_ID")
    required_roles_raw = os.getenv("KEYCLOAK_REQUIRED_ROLES", "")
    required_roles = tuple(role.strip() for role in required_roles_raw.split(",") if role.strip())
    return KeycloakSettings(
        server_url=server_url,
        issuer_url=issuer_url,
        realm=realm,
        client_id=client_id,
        required_roles=required_roles,
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


def get_token_payload(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if not credentials:
        raise _unauthorized("Missing bearer token")

    token = credentials.credentials
    settings = get_keycloak_settings()

    signing_key = get_jwk_client().get_signing_key_from_jwt(token).key

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=settings.issuer,
            options={
                "require": ["exp", "iat", "sub"],
                "verify_aud": False,
            },
        )
    except (InvalidTokenError, PyJWKClientError) as exc:
        raise _unauthorized("Invalid bearer token") from exc

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

    return payload


def require_auth(payload: dict[str, Any] = Depends(get_token_payload)) -> dict[str, Any]:
    settings = get_keycloak_settings()
    if settings.required_roles:
        token_roles = _extract_roles(payload, settings.client_id)
        missing_roles = [role for role in settings.required_roles if role not in token_roles]
        if missing_roles:
            raise _forbidden("Insufficient permissions")
    return payload
