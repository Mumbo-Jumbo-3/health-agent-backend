"""Custom HTTP routes mounted alongside the LangGraph API.

Registered in langgraph.json under the "http" key.
"""

import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from health_agent.auth import require_clerk_user, require_webhook_secret
from health_agent.config import get_settings
from health_agent.db.core import get_session_factory
from health_agent.db.models import SharedConversation, User


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


async def create_share(request: Request) -> JSONResponse:
    try:
        user_id = require_clerk_user(request)
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    thread_id = (body.get("thread_id") or "").strip()
    if not thread_id:
        return JSONResponse({"error": "thread_id required"}, status_code=400)

    title = _truncate(body.get("title") or "", 120)
    first_message = _truncate(body.get("first_message") or "", 500)

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = SharedConversation(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
            first_message=first_message,
        )
        session.add(row)
        session.commit()
        share_id = str(row.share_id)

    return JSONResponse({"share_id": share_id})


async def get_share(request: Request) -> JSONResponse:
    raw_id = request.path_params.get("share_id", "")
    try:
        share_id = uuid.UUID(raw_id)
    except (ValueError, TypeError):
        return JSONResponse({"error": "not found"}, status_code=404)

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = session.get(SharedConversation, share_id)
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        payload = {
            "share_id": str(row.share_id),
            "thread_id": row.thread_id,
            "title": row.title,
            "first_message": row.first_message,
            "created_at": row.created_at.isoformat(),
        }

    return JSONResponse(payload)


def _primary_email(email_addresses: list) -> str | None:
    if not isinstance(email_addresses, list) or not email_addresses:
        return None
    first = email_addresses[0]
    if isinstance(first, dict):
        return first.get("email_address")
    return None


async def sync_user(request: Request) -> JSONResponse:
    try:
        require_webhook_secret(request)
    except HTTPException as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    event_type = body.get("type", "")
    data = body.get("data") or {}
    clerk_user_id = data.get("id")
    if not isinstance(clerk_user_id, str) or not clerk_user_id:
        return JSONResponse({"error": "user id required"}, status_code=400)

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        if event_type == "user.deleted":
            session.query(User).filter(User.clerk_user_id == clerk_user_id).delete()
            session.commit()
            return JSONResponse({"status": "deleted"})

        if event_type in ("user.created", "user.updated"):
            email = _primary_email(data.get("email_addresses") or [])
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            stmt = pg_insert(User).values(
                clerk_user_id=clerk_user_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[User.clerk_user_id],
                set_={
                    "email": stmt.excluded.email,
                    "first_name": stmt.excluded.first_name,
                    "last_name": stmt.excluded.last_name,
                },
            )
            session.execute(stmt)
            session.commit()
            return JSONResponse({"status": "upserted"})

    return JSONResponse({"status": "ignored"})


app = Starlette(
    routes=[
        Route("/share", create_share, methods=["POST"]),
        Route("/share/{share_id}", get_share, methods=["GET"]),
        Route("/internal/users/sync", sync_user, methods=["POST"]),
    ]
)
