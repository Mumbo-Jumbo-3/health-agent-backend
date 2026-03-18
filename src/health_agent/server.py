import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from health_agent.config import get_settings
from health_agent.graph import build_graph
from health_agent.rag.ingest import ingest_resources
from health_agent.rag.retriever import mark_indexed, needs_reindex

sessions: dict[str, list] = {}
compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph

    if needs_reindex(settings):
        print("Indexing resources...")
        result = ingest_resources(settings)
        if result is not None:
            mark_indexed(settings)

    compiled_graph = build_graph(settings)
    yield


settings = get_settings()
app = FastAPI(title="Health Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or uuid.uuid4().hex
    messages = sessions.get(session_id, [])

    messages.append(HumanMessage(content=request.message))
    result = compiled_graph.invoke({"messages": messages})
    messages = result["messages"]
    sessions[session_id] = messages

    ai_message = messages[-1]
    return ChatResponse(response=ai_message.content, session_id=session_id)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or uuid.uuid4().hex
    messages = sessions.get(session_id, [])
    messages.append(HumanMessage(content=request.message))

    async def event_generator():
        full_response = ""
        async for event in compiled_graph.astream_events(
            {"messages": messages}, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    full_response += token
                    yield {"event": "token", "data": token}

        messages.append(AIMessage(content=full_response))
        sessions[session_id] = messages
        yield {"event": "done", "data": json.dumps({"session_id": session_id})}

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    return {"status": "ok"}
