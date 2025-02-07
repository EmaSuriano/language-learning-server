"""CRUD operations"""

from typing import List

from fastapi import HTTPException
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from . import models, schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _user_to_schema(user: models.User) -> schemas.User:
    if user.current_language is None:
        raise HTTPException(status_code=500, detail="User has no language assigned")

    return schemas.User(
        id=user.id,
        email=user.email,
        current_language=schemas.Language(
            id=user.current_language.id,
            code=user.current_language.code,
            name=user.current_language.name,
            has_tts=user.current_language.has_tts,
        ),
        # Convert between model and schema enum
        language_level=schemas.CEFRLevel(user.language_level),
        interests=user.interests,
    )


def get_user(db: Session, user_id: int) -> schemas.User | None:
    user = db.query(models.User).filter(models.User.id == user_id).first()
    return _user_to_schema(user) if user else None


def get_user_by_email(db: Session, email: str) -> schemas.User | None:
    user = db.query(models.User).filter(models.User.email == email).first()
    return _user_to_schema(user) if user else None


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[schemas.User]:
    users = db.query(models.User).offset(skip).limit(limit).all()
    return [_user_to_schema(user) for user in users]


def create_user(db: Session, user: schemas.UserCreate) -> schemas.User:
    # Get the language
    language = (
        db.query(models.Language)
        .filter(models.Language.code == user.language_code.lower())
        .first()
    )

    if language is None:
        raise HTTPException(status_code=404, detail="Language not found")

    # Create new user
    db_user = models.User(
        email=user.email,
        hashed_password=pwd_context.hash(user.password),
        interests=user.interests,  # Now directly stored as a list
        current_language=language,
        language_level=user.language_level,
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return _user_to_schema(db_user)


def update_user(db: Session, user_id: int, user: schemas.UserUpdate) -> schemas.User:
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = user.model_dump(exclude_unset=True)

    # Handle language update
    if "language_code" in update_data:
        language = (
            db.query(models.Language)
            .filter(models.Language.code == update_data["language_code"].lower())
            .first()
        )
        if language is None:
            raise HTTPException(status_code=404, detail="Language not found")
        db_user.current_language = language
        del update_data["language_code"]

    # Update other fields
    for field, value in update_data.items():
        setattr(db_user, field, value)

    db.commit()
    db.refresh(db_user)
    return _user_to_schema(db_user)


def delete_user(db: Session, user_id: int) -> bool:
    result = db.query(models.User).filter(models.User.id == user_id).delete()
    db.commit()
    return bool(result)


def get_languages(
    db: Session, skip: int = 0, limit: int = 100
) -> List[schemas.Language]:
    languages = db.query(models.Language).offset(skip).limit(limit).all()
    return [
        schemas.Language(
            id=lang.id,
            code=lang.code,
            name=lang.name,
            has_tts=lang.has_tts,
        )
        for lang in languages
    ]
