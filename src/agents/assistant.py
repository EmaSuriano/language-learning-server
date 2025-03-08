from collections.abc import AsyncIterator
from enum import Enum
import re
from textwrap import dedent
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from database import schemas
from config import Config


# Define CEFR levels as an Enum for type safety and clarity
class CEFRLevel(str, Enum):
    A1 = "A1 (Beginner)"
    A2 = "A2 (Elementary)"
    B1 = "B1 (Intermediate)"
    B2 = "B2 (Upper Intermediate)"
    C1 = "C1 (Advanced)"
    C2 = "C2 (Proficiency)"


# Get configuration from environment
OLLAMA_MODEL = Config.ollama_model()
OLLAMA_URL = Config.ollama_url()


MAX_RESPONSE_LENGTH = 120
CEFR_LEVELS = [level.value for level in CEFRLevel]


class ConversationContext(BaseModel):
    """'Context of the conversation"""

    situation: schemas.SituationSystem
    user: schemas.User


def _generate_system_prompt(
    context: ConversationContext, role: Literal["human", "ai"]
) -> str:
    cefr_level = CEFR_LEVELS[context.user.language_level - 1]

    communication_guidelines = [
        f"1. Language: Respond only in {context.user.current_language.name}",
        f"2. Proficiency: Use language appropriate for CEFR level {cefr_level}",
        "3. Style: Write natural, conversational responses",
        f"4. Length: Keep responses concise (max {MAX_RESPONSE_LENGTH} characters)",
    ]

    interaction_rules = [
        "- Provide only ONE answer or question per response",
        "- Maintain a natural conversation flow",
        "- Avoid bullet points or lists in responses",
    ]

    if role == "ai":
        communication_guidelines.append(
            f"5. Answer in first person assuming your role of {context.situation.system_role}"
        )

        ai_specific_rules = [
            "- Ask only ONE question per response",
            "- Wait for the user's answer before proceeding",
            "- Guide the conversation according to the scenario",
            "- If the user tries to change the topic, gently guide them back to the scenario",
        ]

        return dedent(
            f"""Role: {context.situation.system_role}

            Scenario:
            {"\n".join(f"• {instruction}" for instruction in context.situation.system_instructions)}

            Communication Guidelines:
            {"\n".join(communication_guidelines)}

            Interaction Rules:
            {"\n".join(interaction_rules + ai_specific_rules)}

            Remember: Your goal is to help the user practice {context.user.current_language.name} while completing the given scenario naturally and effectively.
            """
        )
    else:
        human_specific_rules = [
            "- Focus on the goals that you need to achieve",
            "- Guide the conversation according to the scenario",
        ]

        return dedent(
            f"""Role: You are a student learning {context.user.current_language.name}

            Context: {context.situation.scenario_description}

            Goals:
            {"\n".join(f"• {goal}" for goal in context.situation.user_goals)}

            Interaction Rules:
            {"\n".join(interaction_rules + human_specific_rules)}

            Communication Guidelines:
            {"\n".join(communication_guidelines)}
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
        "{conversation}"

        Specific goal:
        "{goal}"

        IMPORTANT:
        - Provide only 0 or 1 as the output (1 digit only).
        - 0 means the user has NOT achieved the goal.
        - 1 means the user has definitely achieved the goal.
        - Be strict in your evaluation. If there's any doubt, return 0.
        - The goal must be explicitly achieved in the conversation, not just implied or mentioned.
        """),
    )

    llm.num_predict = 1

    # Create an LLM chain
    chain = prompt | llm

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f"{msg.role.upper()}: {msg.content}" for msg in chat_messages]
    )

    # Format the final prompt
    formatted_prompt = prompt.format(
        system_prompt=system_prompt,
        conversation=conversation_text,
        goal=goal,
    )

    # Print the full prompt
    print("Full prompt before AI invocation:\n", formatted_prompt)

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
