from dotenv import load_dotenv
import os

from pydantic import BaseModel
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser


# =========================
# LOAD ENVIRONMENT
# =========================

load_dotenv()


# =========================
# CONTEXT SCHEMA
# =========================

class HiringContext(BaseModel):

    role: Optional[str] = None
    seniority: Optional[str] = None

    technical_skills: List[str] = []
    soft_skills: List[str] = []

    assessment_types: List[str] = []

    remote: Optional[bool] = None
    language: Optional[str] = None

    enough_context: bool = False


# =========================
# OUTPUT PARSER
# =========================

parser = PydanticOutputParser(
    pydantic_object=HiringContext
)


# =========================
# PROMPT
# =========================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You extract hiring requirements from recruiter conversations.

Extract:
- role
- seniority
- technical skills
- soft skills

Set enough_context=true ONLY if:
- role exists
- seniority exists
- at least one technical skill
- at least one soft skill
Return ONLY valid JSON.

{format_instructions}
"""
    ),
    (
        "human",
        "{messages}"
    )
])

prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)


# =========================
# LLM
# =========================

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)


# =========================
# CHAIN
# =========================

chain = prompt | llm | parser


# =========================
# CONTEXT EXTRACTION
# =========================

def extract_context(messages):

    response = chain.invoke({
        "messages": messages
    })

    return response


# =========================
# MISSING FIELD DETECTION
# =========================

def get_missing_fields(context):

    missing = []

    if not context.role:
        missing.append("role")

    if context.role == "Unknown":
        missing.append("role")

    if not context.seniority:
        missing.append("seniority")

    if (not context.technical_skills):
        missing.append("Technical_focus")
    
    if (not context.soft_skills):
        missing.append("assessment_focus")

    return missing


# =========================
# CLARIFICATION QUESTIONS
# =========================

QUESTION_MAP = {
    "role": "What role are you hiring for?",

    "seniority": "What seniority level is the role?",

    "assessment_focus":
        "Are you looking for behavioral, personality, or aptitude assessments?",
    
    "Technical_focus" :
        "Are you looking for someone with in depth knowledge and specialist in the field?"
}


# =========================
# RETRIEVAL QUERY BUILDER
# =========================

def build_retrieval_query(context):

    query = f"""
    {context.role or ""}
    {context.seniority or ""}
    {' '.join(context.technical_skills)}
    {' '.join(context.soft_skills)}
    {' '.join(context.assessment_types)}
    """

    return query.strip()


# =========================
# MAIN ORCHESTRATION
# =========================

def process_conversation(messages):

    context = extract_context(messages)

    missing = get_missing_fields(context)

    if missing:

        reply = QUESTION_MAP[missing[0]]

        return {
            "reply": reply,
            "context": context,
            "retrieval_query": None
        }

    retrieval_query = build_retrieval_query(context)

    return {
        "reply": "Enough information gathered.",
        "context": context,
        "retrieval_query": retrieval_query
    }


# =========================
# TEST
# =========================

messages = [
    {
        "role": "Unknown",
        "content": "hiring a senior full stack developer with good communication skills"
    }
]

response = process_conversation(messages)

print(response)