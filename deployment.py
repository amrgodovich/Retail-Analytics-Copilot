from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from agent.graph_hybrid import calling_graph as graph_app


# MODEL INPUT
class Query(BaseModel):
    id: str
    question: str
    format_hint: str | None = ""


# INIT FASTAPI APP
fastapi_app = FastAPI(
    title="Retail Analytics Copilot",
    version="1.0"
)


# ENDPOINTS
@fastapi_app.get("/health")
def health():
    return {"status": "ok", "message": "Server is working"}

@fastapi_app.post("/ask")
def ask(api_key: str, query: Query):
    """
    Run the LangGraph pipeline.
    """
    state = {
        "id": query.id,
        "question": query.question,
        "format_hint": query.format_hint,
    }

    result = graph_app(api_key,state)

    return {
        "id": result.get("id", query.id),
        "final_answer": result.get("final_answer", ""),
        "explanation": result.get("explanation", ""),
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", 0.0),
        "sql": result.get("sql_query", "")
    }


if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)