"""User router"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import database.db as DB
from database.connection import get_db
from agents.translator import TranslationResponse, translate_text

router = APIRouter(prefix="/translator", tags=["translator"])


@router.get("/", response_model=TranslationResponse)
async def translate_message_get(
    user_id: int, message: str, db: Session = Depends(get_db)
):
    db_user = DB.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # read the target language from the user's profile
    target_language = DB.get_language_by_code(db, code="en")
    if target_language is None:
        raise HTTPException(status_code=404, detail="Target language not found")

    translation = await translate_text(
        content=message,
        source_language=db_user.current_language,
        target_language=target_language,
        level=db_user.language_level,
    )

    # Return structured response
    return translation
