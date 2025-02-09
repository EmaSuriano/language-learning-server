"""Router for the evaluator endpoint"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database.db as DB
from database.connection import get_db
from agents.evaluator import ChatMessage, get_chat_metrics

router = APIRouter(tags=["evaluator"], prefix="/evaluator")


class ReportRequest(BaseModel):
    messages: List[ChatMessage] = [ChatMessage(role="human", content="Hello")]
    user_id: int
    situation_id: int


@router.post("/metrics")
async def metrics(request_data: ReportRequest, db: Session = Depends(get_db)):
    user = DB.get_user(db, request_data.user_id)
    situation = DB.get_situation(db, request_data.situation_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")

    metrics = get_chat_metrics(
        user=user,
        situation=situation,
        chat_messages=request_data.messages,
    )

    return {"metrics": metrics}
