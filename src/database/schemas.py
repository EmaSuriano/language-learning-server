"""Pydantic schemas"""

from enum import IntEnum
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
    interests: List[str] = Field(min_length=1)
    voice_id: str = Field("af_alloy", description="Voice for the TTS service")

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one interest is required")
        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    language_code: Optional[str] = None
    language_level: Optional[CEFRLevel] = None
    interests: Optional[List[str]] = None
    voice_id: Optional[str] = None


class User(UserBase):
    id: int
    current_language: Language
    language_level: CEFRLevel
    voice_id: str | None
    interests: List[str]

    class Config:
        from_attributes = True
