"""Pydantic schemas"""

from enum import IntEnum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class CEFRLevel(IntEnum):
    """Common European Framework of Reference for Languages (CEFR) levels"""

    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


class UserLanguage(BaseModel):
    code: str
    name: str
    level: CEFRLevel


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str
    languages: List[str] = Field(min_length=1)
    interests: List[str] = Field(min_length=1)

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one language is required")
        return v

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one interest is required")
        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    languages: Optional[List[str]] = None
    interests: Optional[List[str]] = None


class User(UserBase):
    id: int
    languages: List[UserLanguage]
    interests: List[str]

    class Config:
        from_attributes = True
