"""CRUD operations"""

from typing import List

from fastapi import HTTPException
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import desc

from models.level_manager import LevelManager, PerformanceMetrics
from config import Config

from . import models, schemas

level_manager = LevelManager(model_path=Config.level_manager_path())
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _user_to_schema(user: models.User) -> schemas.User:
    if user.current_language is None:
        raise HTTPException(status_code=500, detail="User has no language assigned")

    return schemas.User(
        id=user.id,
        email=user.email,
        name=user.name,
        current_language=schemas.Language(
            id=user.current_language.id,
            code=user.current_language.code,
            name=user.current_language.name,
            has_tts=user.current_language.has_tts,
        ),
        # Convert between model and schema enum
        language_level=schemas.CEFRLevel(user.language_level),
        voice_id=user.voice_id,
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


def get_languages(db: Session) -> List[schemas.Language]:
    languages = db.query(models.Language).all()
    return [
        schemas.Language(
            id=lang.id,
            code=lang.code,
            name=lang.name,
            has_tts=lang.has_tts,
        )
        for lang in languages
    ]


def get_language_by_code(db: Session, code: str) -> schemas.Language | None:
    language = db.query(models.Language).filter(models.Language.code == code).first()
    return language


def get_situations(db: Session) -> List[schemas.SituationSystem]:
    situations = db.query(models.Situation).all()
    return [sit for sit in situations]


def get_situation(db: Session, id: int) -> schemas.SituationSystem | None:
    return db.query(models.Situation).filter(models.Situation.id == id).first()


def _session_to_schema(session: models.LearningHistory) -> schemas.LearningHistory:
    """Convert DB model to Pydantic schema"""
    return schemas.LearningHistory(
        id=session.id,
        user_id=session.user_id,
        situation_id=session.situation_id,
        language_id=session.language_id,
        level=schemas.CEFRLevel(session.level),
        grammar_score=session.grammar_score,
        vocabulary_score=session.vocabulary_score,
        fluency_score=session.fluency_score,
        goals_score=session.goals_score,
        level_change=schemas.LevelChangeType(session.level_change),
    )


def _session_to_detail_schema(
    session: models.LearningHistory,
) -> schemas.LearningHistoryDetail:
    """Convert DB model to detailed Pydantic schema with relationships"""
    return schemas.LearningHistoryDetail(
        id=session.id,
        user_id=session.user_id,
        user=_user_to_schema(session.user),
        situation_id=session.situation_id,
        situation=schemas.SituationClient(
            id=session.situation.id,
            name=session.situation.name,
            scenario_description=session.situation.scenario_description,
            user_goals=session.situation.user_goals,
            difficulty=session.situation.difficulty,
        ),
        language_id=session.language_id,
        language=schemas.Language(
            id=session.language.id,
            code=session.language.code,
            name=session.language.name,
            has_tts=session.language.has_tts,
        ),
        level=schemas.CEFRLevel(session.level),
        grammar_score=session.grammar_score,
        vocabulary_score=session.vocabulary_score,
        fluency_score=session.fluency_score,
        goals_score=session.goals_score,
        level_change=schemas.LevelChangeType(session.level_change),
    )


def calculate_level_change(
    db: Session, metrics: PerformanceMetrics, user: schemas.User
) -> str:
    # Get recent sessions for this user (last 5 or fewer)
    recent_sessions = (
        db.query(models.LearningHistory)
        .filter(models.LearningHistory.user_id == user.id)
        .order_by(desc(models.LearningHistory.date))
        .limit(4)
        .all()
    )

    # If there are fewer than 5 sessions, don't run the level manager
    if len(recent_sessions) < 4:
        return schemas.LevelChangeType.MAINTAIN

    # Convert to PerformanceMetrics objects
    metrics_history = [
        PerformanceMetrics(
            s.grammar_score / 100,
            s.vocabulary_score / 100,
            s.fluency_score / 100,
            s.goals_score / 100,
        )
        for s in recent_sessions
    ]

    metrics_history.append(metrics)

    # Get prediction from the level manager
    result = level_manager.predict(metrics_history, user.language_level)

    # Return the decision (increase, maintain, or decrease)
    return schemas.LevelChangeType(result["decision"].upper())


def get_learning_session(
    db: Session, session_id: int
) -> schemas.LearningHistory | None:
    """Get a specific learning session by ID"""
    session = (
        db.query(models.LearningHistory)
        .filter(models.LearningHistory.id == session_id)
        .first()
    )
    return _session_to_schema(session) if session else None


def get_learning_session_detail(
    db: Session, session_id: int
) -> schemas.LearningHistoryDetail | None:
    """Get a detailed learning session by ID with related entities"""
    session = (
        db.query(models.LearningHistory)
        .filter(models.LearningHistory.id == session_id)
        .first()
    )
    return _session_to_detail_schema(session) if session else None


def get_user_learning_sessions(
    db: Session, user: models.User, skip: int = 0, limit: int = 100
) -> List[schemas.LearningHistory]:
    """Get all learning sessions for a specific user"""
    sessions = (
        db.query(models.LearningHistory)
        .filter(models.LearningHistory.user_id == user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [_session_to_schema(session) for session in sessions]


def create_learning_session(
    db: Session, session: schemas.LearningHistoryCreate
) -> schemas.LearningHistory:
    """Create a new learning session"""
    # Verify the user exists
    user = db.query(models.User).filter(models.User.id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify the language exists
    language = (
        db.query(models.Language)
        .filter(models.Language.id == user.current_language.id)
        .first()
    )
    if not language:
        raise HTTPException(status_code=404, detail="Language not found")

    # Verify the situation exists if provided
    if session.situation_id is not None:
        situation = (
            db.query(models.Situation)
            .filter(models.Situation.id == session.situation_id)
            .first()
        )
        if not situation:
            raise HTTPException(status_code=404, detail="Situation not found")

    # Create session model from schema data

    # Create session model from schema data
    session_data = session.model_dump()

    metrics = PerformanceMetrics(
        session.grammar_score / 100,
        session.vocabulary_score / 100,
        session.fluency_score / 100,
        session.goals_score / 100,
    )
    # Calculate level change using the level manager
    level_change = calculate_level_change(db, metrics, user)

    session_data["level_change"] = level_change
    session_data["level"] = user.language_level
    session_data["language_id"] = language.id

    db_session = models.LearningHistory(**session_data)

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return _session_to_schema(db_session)
