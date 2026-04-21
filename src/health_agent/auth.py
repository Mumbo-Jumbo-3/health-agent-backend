"""Clerk JWT verification helpers for Starlette routes."""

from __future__ import annotations

import hmac
import os
from functools import lru_cache

from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from starlette.exceptions import HTTPException
from starlette.requests import Request


class _HeaderAdapter:
    """Wraps Starlette Headers so it satisfies clerk_backend_api.Requestish."""

    def __init__(self, request: Request) -> None:
        self._headers = {k.lower(): v for k, v in request.headers.items()}

    @property
    def headers(self) -> dict[str, str]:
        return self._headers


@lru_cache(maxsize=1)
def _clerk_client() -> Clerk:
    secret = os.environ.get("CLERK_SECRET_KEY")
    if not secret:
        raise RuntimeError("CLERK_SECRET_KEY is not configured")
    return Clerk(bearer_auth=secret)


def require_clerk_user(request: Request) -> str:
    """Verify the Clerk session token on the request and return the Clerk user ID.

    Raises HTTPException(401) if the token is missing or invalid.
    """
    client = _clerk_client()
    state = client.authenticate_request(
        _HeaderAdapter(request),
        AuthenticateRequestOptions(),
    )
    if not state.is_signed_in or not state.payload:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = state.payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_id


def require_webhook_secret(request: Request) -> None:
    """Verify the relay secret sent by the Next.js webhook handler."""
    expected = os.environ.get("BACKEND_WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    header = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        raise HTTPException(status_code=401, detail="Unauthorized")
    provided = header[len(prefix) :]
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
