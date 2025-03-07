import asyncio

from agents.translator import translate_text
import database.db as DB
from database.connection import get_db


async def main():
    db = next(get_db())
    source_language = DB.get_language_by_code(db, "es")
    target_language = DB.get_language_by_code(db, "en")

    original_text = "Hola mi nombre es Pedro"
    print("Original:", original_text)

    translation = await translate_text(
        original_text, source_language, target_language, 1
    )
    print("Translated:", translation.translated_text)


if __name__ == "__main__":
    asyncio.run(main())
