"""User router"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import database.db as DB
from database.connection import get_db
from database import schemas

router = APIRouter(prefix="/situations", tags=["situations"])


# TODO: Add support for multiple languages
@router.get("/", response_model=List[schemas.SituationClient])
async def get_situations(db: Session = Depends(get_db)):
    """Get all situations"""

    return DB.get_situations(db)


@router.get("/{situation_id}", response_model=schemas.SituationClient)
async def read_user(situation_id: int, db: Session = Depends(get_db)):
    """Get a situation by ID"""

    db_situation = DB.get_situation(db, id=situation_id)
    if db_situation is None:
        raise HTTPException(status_code=404, detail="Situation not found")
    return db_situation
