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


class LanguageBase(BaseModel):
    code: str
    name: str
    has_tts: bool = False


class Language(LanguageBase):
    id: int

    class Config:
        from_attributes = True


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


class User(UserBase):
    id: int
    current_language: Language
    language_level: CEFRLevel
    interests: List[str]

    class Config:
        from_attributes = True
