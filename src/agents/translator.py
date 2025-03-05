from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from database import schemas
from config import Config


CEFR_LEVEL = ["A1", "A2", "B1", "B2", "C1", "C2"]


OLLAMA_MODEL = Config.ollama_model()
OLLAMA_URL = Config.ollama_url()

llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0)


async def translate_text(
    content: str,
    target_language: schemas.Language,
    level: int,
) -> str:
    system_prompt = dedent(
        f"""Role: You are a translator, and your task is to translate the following text from English to {target_language.name}.

        User Level: {CEFR_LEVEL[level - 1]}

        IMPORTANT:
        - Provide the translation in the same format as the original text.
        - Use the same style and tone as the original text.
        - Keep the translation concise and clear.
        - Do not add or remove any information.
     """
    )

    prompt = ChatPromptTemplate.from_template(
        dedent("""{system_prompt}

        Text:
        {content}
        """),
    )

    # Create an LLM chain
    chain = prompt | llm

    # Run the chain
    result = await chain.ainvoke(
        {
            "system_prompt": system_prompt,
            "content": content,
        },
    )

    return result
