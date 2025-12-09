from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uuid
import dspy
from agent.graph_hybrid import app as graph_app

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session store
session_store = {}
history_store = {}

def configure_dspy(api_key: str):
    dspy.configure(
        lm=dspy.LM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key
        )
    )


@app.post("/set_key")
def set_key(data: dict):
    api_key = data["api_key"]
    session_id = str(uuid.uuid4())

    session_store[session_id] = {
        "api_key": api_key
    }

    return {"session_id": session_id}


@app.post("/ask")
def ask(data: dict):
    session_id = data["session_id"]
    question = data["question"]

    session = session_store.get(session_id)
    if not session:
        return {"error": "Invalid session, please enter API key again."}

    configure_dspy(session["api_key"])

    result = graph_app.invoke({
        "id": "1",
        "question": question,
    })

    return result
