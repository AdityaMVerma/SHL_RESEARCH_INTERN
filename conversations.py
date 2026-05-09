from dotenv import load_dotenv
import os

from pydantic import BaseModel
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from retrieval import retrieve_assessments


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

    assessment_keys: List[str] = []

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

The conversation may contain multiple turns.
Accumulate hiring requirements across the entire conversation history.
Do not discard previously mentioned requirements unless the user explicitly changes them.

Extract:
- role
- seniority
- technical skills
- soft skills
- relevant SHL assessment categories

Ask question about job whether it is front end heavy or back end heavy or  a specialist roles.

Possible SHL assessment categories include:
- Knowledge & Skills
- Personality & Behavior
- Competencies
- Ability & Aptitude
- Development & 360
- Biodata & Situational Judgment
- Assessment Exercises

Set enough_context=true ONLY if:
- role exists
- seniority exists
- at least one technical skill 
- at least one soft skill exists

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

    if not context.technical_skills:
        missing.append("technical_focus")

    if not context.soft_skills:
        missing.append("behavioral_focus")

    return missing


# =========================
# CLARIFICATION QUESTIONS
# =========================

QUESTION_MAP = {

    "role":
        "What role are you hiring for?",

    "seniority":
        "What seniority level is the role?",

    "technical_focus":
        "What technical skills or technologies should the candidate know well?",

    "behavioral_focus":
        "Are you also looking to assess communication, leadership, or personality traits?"
}


# =========================
# RETRIEVAL QUERY BUILDER
# =========================

def build_retrieval_query(context):

    query = f"""
    Role: {context.role or ""}

    Seniority: {context.seniority or ""}

    Technical Skills:
    {' '.join(context.technical_skills)}

    Soft Skills:
    {' '.join(context.soft_skills)}

    SHL Categories:
    {' '.join(context.assessment_keys)}
    """

    return query.strip()


# =========================
# FORMAT RECOMMENDATIONS
# =========================

def format_recommendations(results):

    formatted = []

    for item in results:

        formatted.append({
            "name": item["name"],
            "url": item["url"],
            "test_type":
                item["keys"][0]
                if item["keys"]
                else "Unknown"
        })

    return formatted


# =========================
# MAIN ORCHESTRATION
# =========================

def process_conversation(messages):

    # Extract accumulated context
    context = extract_context(messages)

    print("\n========== CONTEXT ==========")
    print(context)

    # Determine missing info
    missing = get_missing_fields(context)

    print("\n========== MISSING ==========")
    print(missing)

    # Ask clarification if needed
    if missing:

        reply = QUESTION_MAP[missing[0]]

        return {
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False
        }

    # Build retrieval query
    retrieval_query = build_retrieval_query(context)

    print("\n========== RETRIEVAL QUERY ==========")
    print(retrieval_query)

    # Retrieve assessments
    results = retrieve_assessments(
        retrieval_query,
        top_k=5
    )

    print("\n========== RETRIEVAL RESULTS ==========")

    for r in results:
        print(r["name"])

    # Format recommendations
    recommendations = format_recommendations(results)

    return {
        "reply":
            "Here are some recommended SHL assessments based on your hiring requirements.",

        "recommendations": recommendations,

        "end_of_conversation": False
    }


# =========================
# TEST
# =========================

messages = [
    {
        "role": "user",
        "content": "Need assessments for a Java developer"
    },
    {
        "role": "assistant",
        "content": "What seniority level is the role?"
    },
    {
        "role": "user",
        "content": "Mid-level in node js and angular with stakeholder communication"
    },
    {
        "role": "assistant",
        "content": "Here are some recommendations..."
    },
    {
        "role": "user",
        "content": "Actually also add personality assessments"
    }
]

response = process_conversation(messages)

print("\n========== FINAL RESPONSE ==========")
print(response)