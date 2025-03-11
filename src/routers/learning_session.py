"""User router"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import database.db as DB
import database.schemas as schemas
from database.connection import get_db

router = APIRouter(prefix="/learning-history", tags=["learning-history"])


@router.get("/{session_id}", response_model=schemas.LearningHistory)
async def get_learning_session_by_id(session_id: int, db: Session = Depends(get_db)):
    learning_session = DB.get_learning_session(db, session_id=session_id)

    if learning_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return learning_session


@router.post("/users/{user_id}", response_model=schemas.LearningHistory)
async def add_learning_session(
    session: schemas.LearningHistoryCreate, db: Session = Depends(get_db)
):
    return DB.create_learning_session(db=db, session=session)


@router.get("/users/{user_id}", response_model=List[schemas.LearningHistory])
async def get_learning_sessions_by_user(user_id: int, db: Session = Depends(get_db)):
    user = DB.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return DB.get_user_learning_sessions(db=db, user_id=user_id)


@router.get("/users/{user_id}/progression", response_model=str)
async def get_user_progression(user_id: int, db: Session = Depends(get_db)):
    user = DB.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return DB.get_user_progression(db=db, user=user)
