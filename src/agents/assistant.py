from collections.abc import AsyncIterator
import json
import os
import re
from textwrap import dedent
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from database import schemas

CEFR_LEVEL = ["A1", "A2", "B1", "B2", "C1", "C2"]

load_dotenv()

# Get configuration from environment
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")

assert OLLAMA_MODEL is not None, "OLLAMA_MODEL is not set"
assert OLLAMA_URL is not None, "OLLAMA_URL is not set"


class ConversationContext(BaseModel):
    """'Context of the conversation"""

    situation: schemas.SituationSystem
    user: schemas.User


def _generate_system_prompt(
    context: ConversationContext, role: Literal["human", "ai"]
) -> str:
    cefr_level = CEFR_LEVEL[context.user.language_level - 1]

    if role == "ai":
        return dedent(
            f"""Role: {context.situation.system_role}

            Scenario:
            {"\n".join(f"• {instruction}" for instruction in context.situation.system_instructions)}

            Communication Guidelines:
            1. Language: Respond only in {context.user.current_language.name}
            2. Proficiency: Use language appropriate for CEFR level {cefr_level}
            3. Style: Write natural, conversational responses
            4. Length: Keep responses concise (max 120 characters)
            5. Answer in first person assuming yor role of {context.situation.system_role}

            Interaction Rules:
            - Ask only ONE question per response
            - Wait for the user's answer before proceeding
            - Maintain a natural conversation flow
            - Avoid bullet points or lists in responses
            - Guide the conversation according to the scenario
            - In case the user try to change the topic, gently guide them back to the scenario

            Remember: Your goal is to help the user practice {context.user.current_language.name} while completing the given scenario naturally and effectively.
        """
        )
    else:
        return dedent(
            f"""Role: You are a student learning {context.user.current_language.name}

            Context: {context.situation.scenario_description}

            Goals:
            {"\n".join(f"• {goal}" for goal in context.situation.user_goals)}

            Interaction Rules:
            - Provide only ONE answer or question per response
            - Focus on the goals that you need to achieve
            - Maintain a natural conversation flow
            - Avoid bullet points or lists in responses
            - Guide the conversation according to the scenario

            Communication Guidelines:
            1. Language: Respond only in {context.user.current_language.name}
            2. Proficiency: Use language appropriate for CEFR level {cefr_level}
            3. Style: Write natural, conversational responses
            4. Length: Keep responses concise (max 120 characters)
            """
        )


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ChatMessage(BaseModel):
    role: Literal["human", "ai"]
    content: str


# used for chat communication
chat = ChatOllama(
    model=OLLAMA_MODEL,  # or any model available in your Ollama instance
    temperature=0.2,
    base_url=OLLAMA_URL,  # adjust if your Ollama endpoint is different
)

# used for analysis and reports
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    num_predict=128,
)


async def generate_stream(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
    temperature: float,
) -> AsyncIterator[str | list[str | dict]]:
    """
    Generator function that streams responses from Ollama with proper error handling
    """

    chat.temperature = temperature

    system_prompt = _generate_system_prompt(
        ConversationContext(situation=situation, user=user),
        role="ai",
    )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in chat_messages:
        if msg.role == "human":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    try:
        async for chunk in chat.astream(messages):
            # Each chunk is a BaseMessageChunk; yield its content.
            yield chunk.content
    except ValueError as e:
        yield f"Error: {str(e)}"


def generate_chat_hint(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
    temperature: float,
) -> str | list[str | dict]:
    """
    Get a hint for the user based on the current situation
    """

    chat.temperature = temperature

    system_prompt = _generate_system_prompt(
        ConversationContext(situation=situation, user=user),
        role="human",
    )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in chat_messages:
        if msg.role == "human":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    messages.append(
        HumanMessage(
            content="What can the user answer to continue the conversation? Provide only one answer."
        )
    )

    response = chat.invoke(messages).content

    match = re.search(r'"(.*?)"', str(response))

    # This in case the response is not in first person
    if match:
        return match.group(1)

    return chat.invoke(messages).content


def get_chat_progress(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
) -> list[dict]:
    """
    Get a hint for the user based on the current situation
    """

    system_prompt = _generate_system_prompt(
        ConversationContext(situation=situation, user=user),
        role="human",
    )

    prompt = ChatPromptTemplate.from_template(
        """{system_prompt}

    Conversation:
    {conversation}
    
    Expected output:
    {goals_json}

    Guidelines:
    - Go through the conversation step by step.
    - Pay attention only to the messages from the HUMAN.
    - Provide only the expected output.
    """,
    )

    # Create an LLM chain
    chain = prompt | llm

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f"{msg.role.upper()}: {msg.content}" for msg in chat_messages]
    )

    progress = [{"name": goal, "done": False} for goal in situation.user_goals]

    goals_json = json.dumps(
        progress,
        indent=2,  # Pretty-print for better LLM handling
    )

    # Run the chain
    result = chain.invoke(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
            "goals_json": goals_json,
        },
    )

    print(result)

    json_match = re.search(r"\[.*\]", result, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            parsed_goals = json.loads(json_text)
            return parsed_goals
        except json.JSONDecodeError:
            print("Error: Couldn't parse extracted JSON.")
    else:
        print("Error: No JSON found in response.")

    return progress
