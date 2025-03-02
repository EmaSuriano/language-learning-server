import os
from textwrap import dedent
from typing import Any, AsyncGenerator, Dict, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

from database import schemas
from rag.rag_language_retrieval import LanguageExample, RAGLanguageEvaluator
from agents.translator import translate_text

CEFR_LEVEL = ["A1", "A2", "B1", "B2", "C1", "C2"]


# Get configuration from environment
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")

assert OLLAMA_MODEL is not None, "OLLAMA_MODEL is not set"
assert OLLAMA_URL is not None, "OLLAMA_URL is not set"

llm_feedback = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.5)
llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, num_predict=1)


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


metrics = [
    {"name": "grammar", "description": "Grammatical accuracy"},
    {
        "name": "vocabulary",
        "description": "Appropriate use of vocabulary for level",
    },
    {"name": "fluency", "description": "Natural and smooth communication"},
]


def _generate_system_prompt_new(
    context: ConversationContext,
) -> str:
    cefr_level = CEFR_LEVEL[context.user.language_level - 1]

    return dedent(
        f"""Role: You are a language evaluator, and you are evaluating a conversation between a user and a system.

        User Level: {cefr_level}
        User language: {context.user.current_language.name}
     """
    )


async def _get_example_phrases(
    situation: schemas.SituationSystem,
    rag_evaluator: RAGLanguageEvaluator,
    user: schemas.User,
) -> List[LanguageExample]:
    context = [situation.name, situation.scenario_description, *situation.user_goals]
    examples = rag_evaluator.get_relevant_examples(
        query=" ".join(context),
        level=CEFR_LEVEL[user.language_level - 1],
        k=5,
    )

    for example in examples:
        translated_phrase = await translate_text(
            example.phrase, user.current_language, user.language_level
        )
        example.phrase = translated_phrase

    return examples


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


async def get_chat_report(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
    rag_evaluator: RAGLanguageEvaluator,
) -> AsyncGenerator[str, None]:
    system_prompt = _generate_system_prompt_new(
        ConversationContext(situation=situation, user=user)
    )

    examples = await _get_example_phrases(situation, rag_evaluator, user)

    prompt = ChatPromptTemplate.from_template(
        """{system_prompt}

    Analyze conversation and provide a report for the following topics:

    1. grammar: Grammatical accuracy
    2. vocabulary: Appropriate use of vocabulary for level
    3. fluency: Natural and smooth communication
    4. goals: Achieving the user's goals
    
    Guidelines:
    - Compare the user's language with the provided example phrases for their level
    - Pay special attention to whether they use language appropriate for their level
    - Look for appropriate use of formal/informal language based on the context
    - Remember to keep the same language as the conversation
    - Only return the section of the report that corresponds to the metric

    Conversation:
    {conversation}

    Goals of the user:
    {goals}
    
    Examples of appropriate language for this level:
    {examples}
    """,
    )

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f'"{msg.content}"' for msg in chat_messages if msg.role == "human"]
    )

    examples_text = "\n".join([f'"{msg.phrase}"' for msg in examples])

    goals_text = "\n".join([f"• {goal}" for goal in situation.user_goals])

    # Create an LLM chain
    chain = prompt | llm_feedback

    # Stream the response
    async for chunk in chain.astream(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
            "examples": examples_text,
            "goals": goals_text,
        }
    ):
        yield chunk


async def _get_conversation_metric(
    user: schemas.User,
    situation: schemas.SituationSystem,
    messages: List[ChatMessage],
    examples: List[LanguageExample],
    metric: str,
) -> int:
    system_prompt = _generate_system_prompt_new(
        ConversationContext(situation=situation, user=user)
    )

    prompt = ChatPromptTemplate.from_template(
        dedent("""{system_prompt}
               
        IMPORTANT:
        You MUST return ONLY a number between 0 and 100, with NO additional text.
               
        Rules:
        - Return ONLY a number (0-100)
        - NO explanations
        - NO additional text
        - The number represents the quality score for the metric
        - Higher numbers mean better quality
        - Be critical in your evaluation
               

        Conversation:
        {conversation}
               
        Examples of appropriate language for this level:
        {examples}

        Evaluate and return a single number (0-100) for:
        {metric}
        """),
    )

    # Create an LLM chain
    chain = prompt | llm

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f'"{msg.content}"' for msg in messages if msg.role == "human"]
    )

    examples_text = "\n".join([f'"{msg.phrase}"' for msg in examples])

    # Run the chain
    result = await chain.ainvoke(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
            "examples": examples_text,
            "metric": metric,
        },
    )

    cleaned_result = result.strip().split()[0]  # Take first word only
    return int(cleaned_result)


async def get_overview(
    user: schemas.User,
    situation: schemas.SituationSystem,
    messages: List[ChatMessage],
    rag_evaluator: RAGLanguageEvaluator,
) -> ConversationMetrics:
    examples = await _get_example_phrases(situation, rag_evaluator, user)

    results = {}

    for metric in metrics:
        result = await _get_conversation_metric(
            user=user,
            situation=situation,
            messages=messages,
            examples=examples,
            metric=f"{metric['name']}: {metric['description']}",
        )

        results[metric["name"]] = result

    return ConversationMetrics(**results)
