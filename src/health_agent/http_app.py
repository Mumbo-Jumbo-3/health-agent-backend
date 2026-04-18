"""Custom HTTP routes mounted alongside the LangGraph API.

Registered in langgraph.json under the "http" key.
"""

import os
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from health_agent.config import get_settings
from health_agent.db.core import get_session_factory
from health_agent.db.models import SharedConversation


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


async def create_share(request: Request) -> JSONResponse:
    expected_key = os.environ.get("LANGSMITH_API_KEY", "")
    provided_key = request.headers.get("x-api-key", "")
    if not expected_key or provided_key != expected_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

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


app = Starlette(
    routes=[
        Route("/share", create_share, methods=["POST"]),
        Route("/share/{share_id}", get_share, methods=["GET"]),
    ]
)
