"""User router"""

from typing import List

from fastapi import APIRouter, Depends
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
