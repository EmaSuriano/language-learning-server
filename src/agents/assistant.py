from collections.abc import AsyncIterator
import os
import re
from textwrap import dedent
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from database import schemas

CEFR_LEVEL = ["A1", "A2", "B1", "B2", "C1", "C2"]

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


print(OLLAMA_MODEL)

# used for chat communication
chat = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
    base_url=OLLAMA_URL,
)

# used for analysis and reports
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
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

    full_response = ""
    try:
        async for chunk in chat.astream(messages):
            content = chunk.content

            if isinstance(content, str):
                full_response += content
            elif isinstance(content, list):
                full_response += "".join(str(item) for item in content)

            # Check if a horizontal line appears in the accumulated response
            if "---" in full_response:
                # Only yield the content before the horizontal line
                cleaned_response = full_response.split("---")[0].strip()
                # Only yield if we haven't already yielded this exact content
                if cleaned_response != full_response:
                    yield cleaned_response
                # Stop the streaming
                return

            # Otherwise continue streaming normally
            yield content
    except ValueError as e:
        yield f"Error: {str(e)}"


async def generate_chat_hint(
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

    response = await chat.ainvoke(messages)

    match = re.search(r'"(.*?)"', str(response.content))

    # This in case the response is not in first person
    if match:
        return match.group(1)

    return response.content


async def _get_goal_progress(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
    goal: str,
) -> bool:
    system_prompt = _generate_system_prompt(
        ConversationContext(situation=situation, user=user),
        role="human",
    )

    prompt = ChatPromptTemplate.from_template(
        dedent("""{system_prompt}

        Conversation:
        {conversation}

        User goal:
        {goal}

        IMPORTANT:
        - Provide only 0 or 1 as the ouput.
        - 0 means the user has not achieved the goal.
        - 1 means the user has achieved the goal.
        """),
    )

    llm.num_predict = 1

    # Create an LLM chain
    chain = prompt | llm

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f"{msg.role.upper()}: {msg.content}" for msg in chat_messages]
    )

    # Run the chain
    result = await chain.ainvoke(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
            "goal": goal,
        },
    )

    return result == "1"


async def _get_conversation_over(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
) -> bool:
    system_prompt = _generate_system_prompt(
        ConversationContext(situation=situation, user=user),
        role="ai",
    )

    prompt = ChatPromptTemplate.from_template(
        dedent("""{system_prompt}

        Conversation:
        {conversation}

        Identify if the conversation has ended.

        IMPORTANT:
        - Provide only 0 or 1 as the ouput.
        - 0 means the conversation is not over.
        - 1 means the conversation is over.
        """),
    )

    llm.num_predict = 1

    # Create an LLM chain
    chain = prompt | llm

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f"{msg.role.upper()}: {msg.content}" for msg in chat_messages]
    )

    # Run the chain
    result = await chain.ainvoke(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
        },
    )

    return result == "1"


async def get_chat_progress(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
) -> Dict[str, Any]:
    """
    Get a hint for the user based on the current situation
    """

    goals = [{"name": goal, "done": False} for goal in situation.user_goals]

    for goal in goals:
        # Run the chain
        result = await _get_goal_progress(
            user=user,
            situation=situation,
            chat_messages=chat_messages,
            goal=str(goal["name"]),
        )

        goal["done"] = result

    conversation_over = await _get_conversation_over(
        user=user,
        situation=situation,
        chat_messages=chat_messages,
    )

    return {"goals": goals, "conversation_over": conversation_over}
