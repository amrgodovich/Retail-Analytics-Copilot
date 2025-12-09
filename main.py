import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uuid
import dspy
import asyncio
from datetime import datetime
from agent.graph_hybrid import app as graph_app
from agent.summarizer_module import summarizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = {} # for api keys and other session data etc etc
history_store = {} # to store conversation histories per session

# def configure_dspy(api_key: str):
#     dspy.configure(
#         lm=dspy.LM(
#             model="gemini/gemini-2.5-flash",
#             api_key=api_key
#         )
#     )

dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash", api_key="dummy"))

def generate_question_id(session_id):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%y_%H-%M-%S")
    return f"{session_id}_{timestamp}"

async def update_history(session_id: str, question: str, result: str, api_key: str):
    history = history_store.get(session_id, "")

    try:
        with dspy.context(lm=dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)):
            new_history = summarizer(old_history=history, question=question, result=result)
            history_store[session_id] = new_history['new_history']
    except Exception as e:
        print("error: The history store didnt work as the API key might be invalid or exceeded its quota.")
        return {"error": "The history store didnt work as the API might be invalid or exceeded its quota."}
    
    print(f"history of {session_id} is updated!")

@app.post("/set_key")
def set_key(data: dict):
    api_key = data["api_key"]
    session_id = str(uuid.uuid4())
    session_store[session_id] = {"api_key": api_key}
    return {"session_id": session_id}

@app.post("/ask")
async def ask(data: dict):
    session_id = data["session_id"]
    question = data["question"]
    
    session = session_store.get(session_id)
    if not session:
        return {"error": "Invalid session, please enter API key again."}
    
    api_key = session["api_key"]
    history = history_store.get(session_id, "")
    
    question_id = generate_question_id(session_id)
    
    try:
        with dspy.context(lm=dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)):
            result = graph_app.invoke({
                "id": question_id,
                "question": question,
                "history": history,
            })
    except Exception as e:
        print("error: your API key might be invalid or exceeded its quota.")
        return {"error": "your API key might be invalid or exceeded its quota."}

    asyncio.create_task(update_history(session_id, question, result, api_key))
    
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)