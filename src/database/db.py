"""CRUD operations"""

from typing import List

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from . import models, schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _user_to_schema(user: models.User) -> schemas.User:
    return schemas.User(
        id=int(user.id),
        email=str(user.email),
        languages=[
            schemas.UserLanguage(
                code=lang.code,
                name=lang.name,
                level=schemas.CEFRLevel(getattr(lang, "level", 1)),
            )
            for lang in user.languages
        ],
        interests=[str(i.interest) for i in user.interests],
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
    db_user = models.User(
        email=user.email, hashed_password=pwd_context.hash(user.password)
    )
    db.add(db_user)

    for interest_name in user.interests:
        interest = (
            db.query(models.Interest)
            .filter(models.Interest.interest == interest_name.lower())
            .first()
        )
        if interest:
            db_user.interests.append(interest)

    for language_code in user.languages:
        language = (
            db.query(models.Language)
            .filter(models.Language.code == language_code.lower())
            .first()
        )
        if language:
            db_user.languages.append(language)

    db.commit()
    db.refresh(db_user)
    return _user_to_schema(db_user)


def delete_user(db: Session, user_id: int) -> bool:
    result = db.query(models.User).filter(models.User.id == user_id).delete()
    db.commit()
    return bool(result)
