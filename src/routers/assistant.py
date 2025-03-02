"""Router for the assistant endpoint"""

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database.db as DB
from database.connection import get_db
from agents.assistant import (
    ChatMessage,
    generate_stream,
    generate_chat_hint,
    get_chat_progress,
)

router = APIRouter(tags=["assistant"], prefix="/assistant")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = [ChatMessage(role="human", content="Hello")]
    user_id: int
    situation_id: int
    temperature: float = 0.2


class ReportRequest(BaseModel):
    messages: List[ChatMessage] = [ChatMessage(role="human", content="Hello")]
    user_id: int
    situation_id: int


@router.post("/chat")
async def chat(
    request_data: ChatRequest, request: Request, db: Session = Depends(get_db)
):
    """
    Endpoint that streams chat responses from Ollama with proper headers
    """

    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    async def event_stream():
        try:
            async for chunk in generate_stream(
                user=user,
                situation=situation,
                chat_messages=request_data.messages,
                temperature=request_data.temperature,
            ):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
        media_type="text/event-stream",
    )


@router.post("/chat/no-stream")
async def chat_full(request_data: ChatRequest, db: Session = Depends(get_db)):
    """
    Non-streaming chat response that returns the full content at once.
    """

    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    full_content = []
    try:
        async for chunk in generate_stream(
            user=user,
            situation=situation,
            chat_messages=request_data.messages,
            temperature=request_data.temperature,
        ):
            full_content.append(chunk)
    except ValueError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    flattened_content = [
        item if isinstance(item, str) else json.dumps(item)
        for sublist in full_content
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]

    return {"content": "".join(flattened_content)}


@router.post("/hint")
async def chat_hint(request_data: ChatRequest, db: Session = Depends(get_db)):
    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    chat_hint = await generate_chat_hint(
        user=user,
        situation=situation,
        chat_messages=request_data.messages,
        temperature=request_data.temperature,
    )

    return {"hint": chat_hint}


@router.post("/chat/progress")
async def chat_progress(request_data: ChatRequest, db: Session = Depends(get_db)):
    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    chat_progress = await get_chat_progress(
        user=user,
        situation=situation,
        chat_messages=request_data.messages,
    )

    return chat_progress
