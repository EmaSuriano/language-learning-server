"""Router for the evaluator endpoint"""

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database.db as DB
from database.connection import get_db
from agents.evaluator import ChatMessage, get_chat_report, get_overview
from rag.connection import get_rag_evaluator
from rag.rag_language_retrieval import RAGLanguageEvaluator

router = APIRouter(tags=["evaluator"], prefix="/evaluator")


class ReportRequest(BaseModel):
    messages: List[ChatMessage] = [ChatMessage(role="human", content="Hello")]
    user_id: int
    situation_id: int


@router.post("/report")
async def generate_report(
    request_data: ReportRequest,
    db: Session = Depends(get_db),
    rag_evaluator: RAGLanguageEvaluator = Depends(get_rag_evaluator),
):
    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    async def event_stream():
        try:
            async for chunk in get_chat_report(
                user=user,
                situation=situation,
                chat_messages=request_data.messages,
                rag_evaluator=rag_evaluator,
            ):
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


@router.post("/overview")
async def generate_overview(
    request_data: ReportRequest,
    db: Session = Depends(get_db),
    rag_evaluator: RAGLanguageEvaluator = Depends(get_rag_evaluator),
):
    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    overview = await get_overview(
        user=user,
        situation=situation,
        messages=request_data.messages,
        rag_evaluator=rag_evaluator,
    )

    return overview
