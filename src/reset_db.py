"""Database initialization"""

import json
from sqlalchemy.orm import Session

from database.connection import engine
from database.models import Base, Language, User, Situation
import os

# Taken from kokoro-tts config
TTS_LANGUAGES = {"en", "es", "fr", "hi", "it", "pt", "ja", "zh"}

# Subset of most used langugages
ALL_LANGUAGES = [
    ("ar", "Arabic"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("zh", "Chinese"),
]

DEFAULT_LANGUAGE = "en"


def reset_db():
    """Reset the database"""

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    session = Session(engine)

    file_path = os.path.join(os.path.dirname(__file__), "persistent", "situations.json")
    with open(file_path, "r") as file:
        situations = json.load(file)  # load = load file
        for situation in situations:
            session.add(Situation(**situation))

    for code, name in ALL_LANGUAGES:
        language = Language(code=code, name=name, has_tts=(code in TTS_LANGUAGES))

        if code == DEFAULT_LANGUAGE:
            current_language = language

        session.add(language)

    session.add(
        User(
            email="example@mail.com",
            hashed_password="hashed_password",
            current_language=current_language,
            language_level=1,
            voice_id="af_alloy",
            name="John Doe",
        )
    )

    session.commit()
    session.close()


if __name__ == "__main__":
    print("Creating database tables...")
    reset_db()
    print("Done!")
