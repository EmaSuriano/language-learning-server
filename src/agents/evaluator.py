import asyncio
from collections.abc import AsyncIterator
import json
import re
from textwrap import dedent
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


from database import schemas
from rag.rag_language_retrieval import LanguageExample, RAGLanguageEvaluator

CEFR_LEVEL = ["A1", "A2", "B1", "B2", "C1", "C2"]
OLLAMA_URL = "http://localhost:11434"


class ConversationContext(BaseModel):
    """'Context of the conversation"""

    situation: schemas.SituationSystem
    user: schemas.User


class ChatMessage(BaseModel):
    role: Literal["human", "ai"]
    content: str


class ConversationMetrics(BaseModel):
    grammar: int = 50
    vocabulary: int = 50
    fluency: int = 50


def _generate_system_prompt(
    context: ConversationContext,
    rag_evaluator: RAGLanguageEvaluator,
) -> str:
    cefr_level = CEFR_LEVEL[context.user.language_level - 1]

    return dedent(
        f"""Role: You are a language evaluator, and you are evaluating a conversation between a user and a system.

        Communication Guidelines:
            1. Language: Respond only in {context.user.current_language.name}
            2. Report: Provide a detailed report on the user's language usage
            3. Provide feedback when necessary
            4. Be professional and respectful

        User Level: {cefr_level}
        
        Context: {context.situation.scenario_description}

        Goals of the user:
        {"\n".join(f"• {goal}" for goal in context.situation.user_goals)}
     """
    )


# used for analysis and reports
llm = OllamaLLM(model="phi4", base_url=OLLAMA_URL, temperature=0)


async def get_chat_report(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
    rag_evaluator: RAGLanguageEvaluator,
) -> AsyncGenerator[str, None]:
    context = ConversationContext(situation=situation, user=user)
    system_prompt = _generate_system_prompt(context, rag_evaluator)

    relevant_examples: List[LanguageExample] = []
    for msg in chat_messages:
        if msg.role == "human":
            examples = rag_evaluator.get_relevant_examples(
                query=msg.content,
                level=CEFR_LEVEL[user.language_level - 1],
                k=3,
            )
            relevant_examples.extend(examples)

    if relevant_examples:
        system_prompt += "\n\nRelevant examples for this level and context:\n"
        for example in relevant_examples:
            system_prompt += f"• {example.phrase} (Context: {example.context})\n"

    print(system_prompt)

    prompt = ChatPromptTemplate.from_template(
        """{system_prompt}

    Conversation:
    {conversation}

    Analyze conversation and provide a report for the following topics:

    1. grammar: Grammatical accuracy
    2. vocabulary: Appropriate use of vocabulary for level
    3. fluency: Natural and smooth communication
    
    Guidelines:
    - Compare the user's language with the provided example phrases for their level
    - Pay special attention to whether they use language appropriate for their level
    - To evaluate give the result pay attention at the HUMAN messages
    - Look for appropriate use of formal/informal language based on the context
    - Remember to keep the same language as the conversation
    """,
    )

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f"{msg.role.upper()}: {msg.content}" for msg in chat_messages]
    )

    # Create an LLM chain
    chain = prompt | llm

    # Stream the response
    async for chunk in chain.astream(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
        }
    ):
        yield chunk
