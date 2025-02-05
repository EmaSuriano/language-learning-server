"""Database models"""

import enum

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class CEFRLevel(enum.IntEnum):
    """Common European Framework of Reference for Languages (CEFR) levels"""

    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


user_interests = Table(
    "user_interests",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("interest_id", Integer, ForeignKey("interests.id")),
)

user_languages = Table(
    "user_languages",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("language_id", Integer, ForeignKey("languages.id")),
    Column("level", Integer, default=1),  # CEFRLevel as integer
)


class Interest(Base):
    """Interest model for users"""

    __tablename__ = "interests"
    id = Column(Integer, primary_key=True)
    interest = Column(String, unique=True)


class Language(Base):
    """Language model"""

    __tablename__ = "languages"
    id = Column(Integer, primary_key=True)
    code = Column(String(5), unique=True)
    name = Column(String, unique=True)
    has_tts = Column(Boolean, default=False)


class User(Base):
    """User model with interests"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    languages = relationship("Language", secondary=user_languages)
    user_interests = relationship("Interest", secondary=user_interests)
