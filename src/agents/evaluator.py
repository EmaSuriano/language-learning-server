from collections.abc import AsyncIterator
import json
import re
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


from database import schemas

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


def _generate_system_prompt(context: ConversationContext) -> str:
    cefr_level = CEFR_LEVEL[context.user.language_level - 1]

    return dedent(
        f"""Role: You are a language evaluator, and you are evaluating a conversation between a user and a system.        

        Language: {context.user.current_language.name}

        User Level: {cefr_level}
        
        Context: {context.situation.scenario_description}

        Goals of the user:
        {"\n".join(f"• {goal}" for goal in context.situation.user_goals)}
     """
    )


# used for analysis and reports
llm = OllamaLLM(model="phi4", base_url=OLLAMA_URL, temperature=0)


def get_chat_metrics(
    user: schemas.User,
    situation: schemas.SituationSystem,
    chat_messages: List[ChatMessage],
    rag_evaluator: Optional[RAGLanguageEvaluator] = None,
) -> ConversationMetrics:
    system_prompt = _generate_system_prompt(
        ConversationContext(situation=situation, user=user),
    )

    relevant_examples = []
    for msg in chat_messages:
        if msg.role == "human":
            examples = rag_evaluator.get_relevant_examples(
                query=msg.content,
                level=CEFR_LEVEL[user.language_level - 1],
                category=situation.category,
                k=3,
            )
            relevant_examples.extend(examples)

    if relevant_examples:
        system_prompt += "\n\nRelevant examples for this level and context:\n"
        for example in relevant_examples:
            system_prompt += f"• {example.phrase} (Context: {example.context})\n"

    prompt = ChatPromptTemplate.from_template(
        """{system_prompt}

    Conversation:
    {conversation}

    Analyze this language learning conversation and provide scores (0-100) for:

    1. grammar: Grammatical accuracy
    2. vocabulary: Appropriate use of vocabulary for level
    3. fluency: Natural and smooth communication
    

    Guidelines:
    - Go through the conversation step by step.
    - Compare the user's language with the provided example phrases for their level
    - Pay special attention to whether they use language appropriate for their level
    - To evaluate give the result pay attention at the HUMAN messages
    - Consider if they're using simpler or more complex language than expected for their level
    - Look for appropriate use of formal/informal language based on the context
    - Provide only the expected output
    - Be strict with the evaluation

    Expected output:
    {metrics_json}
    """,
    )

    # Format the conversation as a single string
    conversation_text = "\n".join(
        [f"{msg.role.upper()}: {msg.content}" for msg in chat_messages]
    )

    metrics = ConversationMetrics()

    goals_json = json.dumps(
        metrics.model_dump_json(),
        indent=2,  # Pretty-print for better LLM handling
    )

    llm.num_predict = 40

    # Create an LLM chain
    chain = prompt | llm

    # Run the chain
    result = chain.invoke(
        {
            "system_prompt": system_prompt,
            "conversation": conversation_text,
            "metrics_json": goals_json,
        },
    )

    print(result)

    json_match = re.search(r"\{.*\}", result, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            parsed_goals = json.loads(json_text)
            return parsed_goals
        except json.JSONDecodeError:
            print("Error: Couldn't parse extracted JSON.")
    else:
        print("Error: No JSON found in response.")

    return metrics
