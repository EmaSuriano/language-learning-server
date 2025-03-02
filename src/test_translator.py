import asyncio

from agents.translator import translate_text
import database.db as DB
from database.connection import get_db


async def main():
    db = next(get_db())
    language = DB.get_language_by_code(db, "en")

    original_text = "Hola mi nombre es Pedro"
    print("Original:", original_text)

    translated_text = await translate_text(original_text, language, 1)
    print("Translated:", translated_text)


if __name__ == "__main__":
    asyncio.run(main())
