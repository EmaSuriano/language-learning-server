"""User router"""

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import database.db as DB
from database.connection import get_db
from database import schemas

router = APIRouter(prefix="/languages", tags=["languages"])


@router.get("/", response_model=List[schemas.Language])
async def get_languages(db: Session = Depends(get_db)):
    """Get all languages"""

    return DB.get_languages(db)
