"""Router for the assistant endpoint"""

import asyncio
import json
from collections.abc import AsyncIterator
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Set

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

router = APIRouter(tags=["assistant"], prefix="/assistant")


CEFR_LEVEL = ["A1", "A2", "B1", "B2", "C1", "C2"]
OLLAMA_URL = "http://localhost:11434"


class ConversationContext(BaseModel):
    """'Context of the conversation"""

    current_topic: str
    user_level: int  # CEFR level: 1=A1, 2=A2, 3=B1, 4=B2, 5=C1, 6=C2
    recent_vocabulary: Set[str]
    grammar_patterns: Set[str]
    user_interests: List[str]
    target_language: str = "English"


def _generate_system_prompt(context: ConversationContext) -> str:
    cefr_level = CEFR_LEVEL[context.user_level - 1]

    return dedent(
        f"""You are a helpful language learning partner having a natural conversation. 

            Situation:
                - You are a seller in a coffe shop and the customer is asking about the menu.
                - Engage with the customer and provide information about the menu.
                - Ask him for different options of coffee and pastries.
                - Ask him if he wants to sit in or take away.
                - Ask him if he wants to pay by cash or card.
                - Ask him if he wants to add a tip.
                - Ask him if he wants a receipt.
                - Greet the user and thank him for his visit.

            Guidelines:
                - Keep responses concise and engaging, no more than a tweet length
                - Stay in context of the conversation
                - You can only answer back in {context.target_language}
                - Adapt your language to CEFR level {cefr_level}.
            """
    )


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message] = [Message(role="user", content="Hello")]
    language: str = "English"
    temperature: float = 0.3


async def generate_stream(
    request_data: ChatRequest,
) -> AsyncIterator[str | list[str | dict], None]:
    """
    Generator function that streams responses from Ollama with proper error handling
    """
    system_prompt = _generate_system_prompt(
        ConversationContext(
            current_topic="general",
            user_level=1,
            recent_vocabulary=set(),
            grammar_patterns=set(),
            user_interests=["music", "movies"],
            target_language=request_data.language,
        )
    )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    for msg in request_data.messages:
        if msg.role.lower() == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    llm = ChatOllama(
        model="phi4",  # or any model available in your Ollama instance
        temperature=request_data.temperature,
        base_url=OLLAMA_URL,  # adjust if your Ollama endpoint is different
    )

    # If tool definitions are provided, bind them to the model
    # if request_data.tools:
    #     llm = llm.bind_tools(request_data.tools)

    try:
        async for chunk in llm.astream(messages):
            # Each chunk is a BaseMessageChunk; yield its content.
            yield chunk.content
            await asyncio.sleep(0.01)  # mimic a small pause between chunks if desired
    except Exception as e:
        yield f"Error: {str(e)}"


@router.post("/chat")
async def chat(request_data: ChatRequest, request: Request):
    """
    Endpoint that streams chat responses from Ollama with proper headers
    """

    async def event_stream():
        try:
            async for chunk in generate_stream(request_data):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
        media_type="text/event-stream",
    )


@router.post("/chat/no-stream")
async def chat_full(request_data: ChatRequest):
    """
    Non-streaming chat response that returns the full content at once.
    """
    full_content = []
    try:
        async for chunk in generate_stream(request_data):
            full_content.append(chunk)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"content": "".join(full_content)}
