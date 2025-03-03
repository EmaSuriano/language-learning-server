"""User router"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import database.db as DB
import database.schemas as schemas
from database.connection import get_db

router = APIRouter(prefix="/learning-history", tags=["learning-history"])


@router.post("/", response_model=schemas.LearningHistory)
async def add_learning_session(
    session: schemas.LearningHistoryBase, db: Session = Depends(get_db)
):
    return DB.create_learning_session(db=db, session=session)


@router.get("/{session_id}", response_model=schemas.LearningHistory)
async def get_by_id(session_id: int, db: Session = Depends(get_db)):
    """Get learning session by id"""
    learning_session = DB.get_learning_session(db, session_id=session_id)

    if learning_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return learning_session


@router.get("/users/{user_id}", response_model=List[schemas.LearningHistory])
async def read_user(user_id: int, db: Session = Depends(get_db)):
    """Get a user by ID"""

    user = DB.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return DB.get_user_learning_sessions(db=db, user=user)
