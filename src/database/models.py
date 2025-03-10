"""Database models"""

from datetime import datetime
import enum
import json
from typing import List, Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


class ArrayType(TypeDecorator):
    """Custom type to store arrays as JSON strings"""

    impl = Text
    cache_ok = True

    def process_bind_param(
        self, value: Optional[List[str]], dialect: Any
    ) -> Optional[str]:
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value: Optional[str], dialect: Any) -> List[str]:
        if value is not None:
            return json.loads(value)
        return []


class CEFRLevel(enum.IntEnum):
    """Common European Framework of Reference for Languages (CEFR) levels"""

    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


class Language(Base):
    """Language model"""

    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(5), unique=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    has_tts: Mapped[bool] = mapped_column(Boolean, default=False)


class Situation(Base):
    """Situation for roleplay model"""

    __tablename__ = "situations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    scenario_description: Mapped[str] = mapped_column(String)
    user_goals: Mapped[List[str]] = mapped_column(ArrayType)
    system_role: Mapped[str] = mapped_column(String)
    difficulty: Mapped[str] = mapped_column(String)
    system_instructions: Mapped[List[str]] = mapped_column(ArrayType)


class User(Base):
    """User model with current language"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    hashed_password: Mapped[str] = mapped_column(String)

    # One-to-one relationship with Language
    current_language_id: Mapped[int] = mapped_column(ForeignKey("languages.id"))
    current_language: Mapped[Language] = relationship("Language")
    voice_id: Mapped[str | None] = mapped_column(String)

    # Store CEFR level directly in user
    language_level: Mapped[CEFRLevel] = mapped_column(Integer, default=CEFRLevel.A1)


class LevelChangeType(enum.Enum):
    """Level change type for learning sessions"""

    INCREASE = "INCREASE"
    MAINTAIN = "MAINTAIN"
    DECREASE = "DECREASE"


class LearningHistory(Base):
    """Learning session history model"""

    __tablename__ = "learning_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    situation_id: Mapped[int] = mapped_column(ForeignKey("situations.id"))
    language_id: Mapped[int] = mapped_column(ForeignKey("languages.id"))
    level: Mapped[CEFRLevel] = mapped_column(Integer)
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped[User] = relationship("User")
    situation: Mapped[Situation] = relationship("Situation")
    language: Mapped[Language] = relationship("Language")

    # Performance metrics
    grammar_score: Mapped[float] = mapped_column(Float)
    vocabulary_score: Mapped[float] = mapped_column(Float)
    fluency_score: Mapped[float] = mapped_column(Float)
    goals_score: Mapped[float] = mapped_column(Float)

    # Level change recommendation
    level_change: Mapped[str] = mapped_column(
        String, default=LevelChangeType.MAINTAIN.value
    )
