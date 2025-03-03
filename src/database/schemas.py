"""Pydantic schemas"""

from datetime import datetime
from enum import IntEnum, Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class CEFRLevel(IntEnum):
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


class Language(BaseModel):
    id: int
    code: str
    name: str
    has_tts: bool = False

    class Config:
        from_attributes = True


class BaseSituation(BaseModel):
    id: int
    name: str = Field(description="Name of the situation")

    class Config:
        from_attributes = True


class SituationClient(BaseSituation):
    scenario_description: str = Field(description="Description of the scenario")
    user_goals: List[str] = Field(min_length=1, description="Goals for the user")
    difficulty: str = Field(description="Difficulty level of the situation")


class SituationSystem(SituationClient):
    system_role: str = Field(description="Role for the system prompt")
    system_instructions: List[str] = Field(
        min_length=1, description="Instructions for the situation"
    )


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str
    language_code: str = Field(
        "en", description="Language code the user wants to learn"
    )
    language_level: CEFRLevel = Field(
        CEFRLevel.A1, description="CEFR level of the user"
    )
    voice_id: str = Field("af_alloy", description="Voice for the TTS service")


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    language_code: Optional[str] = None
    language_level: Optional[CEFRLevel] = None
    voice_id: Optional[str] = None


class User(UserBase):
    id: int
    current_language: Language
    language_level: CEFRLevel
    voice_id: str | None

    class Config:
        from_attributes = True


class LevelChangeType(str, Enum):
    INCREASE = "INCREASE"
    MAINTAIN = "MAINTAIN"
    DECREASE = "DECREASE"


# Learning Session schemas
class LearningHistoryBase(BaseModel):
    user_id: int = Field(..., description="ID of the user")
    situation_id: Optional[int] = Field(
        None, description="ID of the situation used in the session"
    )
    language_id: int = Field(..., description="ID of the language being learned")
    level: CEFRLevel = Field(..., description="CEFR level of the session")
    date: datetime = Field(
        default_factory=datetime.utcnow, description="Date of the learning session"
    )

    # Performance metrics
    grammar_score: float = Field(..., ge=0, le=100, description="Grammar score (0-100)")
    vocabulary_score: float = Field(
        ..., ge=0, le=100, description="Vocabulary score (0-100)"
    )
    fluency_score: float = Field(..., ge=0, le=100, description="Fluency score (0-100)")
    goals_score: float = Field(
        ..., ge=0, le=100, description="Goals achievement score (0-100)"
    )


class LearningHistoryCreate(LearningHistoryBase):
    pass


class LearningHistory(LearningHistoryBase):
    id: int
    # Level change recommendation calculated by the LevelManager
    level_change: LevelChangeType = Field(
        LevelChangeType.MAINTAIN, description="Level change recommendation"
    )

    class Config:
        from_attributes = True


class LearningHistoryDetail(LearningHistory):
    user: User
    situation: SituationClient
    language: Language

    class Config:
        from_attributes = True
