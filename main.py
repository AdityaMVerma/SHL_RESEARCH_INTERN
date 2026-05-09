from fastapi import FastAPI
from pydantic import BaseModel

from conversations import process_conversation


app = FastAPI()


# =========================
# REQUEST SCHEMA
# =========================

class ChatRequest(BaseModel):
    messages: list


# =========================
# HEALTH ENDPOINT
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================
# CHAT ENDPOINT
# =========================

@app.post("/chat")
def chat(request: ChatRequest):

    response = process_conversation(
        request.messages
    )

    return response