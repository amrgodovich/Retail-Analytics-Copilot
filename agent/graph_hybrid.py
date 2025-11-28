from llm_setup import load_llm
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, Optional
from rag.retrieval import BM25Retriever
from tools.sqlite_tool import SQLiteTool
from dspy_signatures import RouterModule
from dspy_signatures import PlannerModule



load_llm()   
router = RouterModule()
planner = PlannerModule()
db = SQLiteTool("data/northwind.sqlite")
retriever = BM25Retriever(docs_folder="docs")


class AgentState(dict):
    question: str
    mode: str
    planner_output: dict
    rag_chunks: list
    sql_query: str
    sql_result: dict
    final_answer: dict
    retries: int



def router_node(state: AgentState):
    """
        rag | sql | hybrid 
    """
    question = state["question"]
    result = router(question)
    state["mode"] = result["mode"]
    if state["mode"] not in ["rag", "sql", "hybrid"]:
            state["mode"] = "hybrid"
    return state

def retriever_node(state: AgentState):
    mode = state.get("mode", "rag")

    if mode not in ["rag", "hybrid"]:
        state["rag_chunks"] = []
        return state

    question = state["question"]
    chunks = retriever.retrieve(question, k=5)

    state["rag_chunks"] = chunks
    return state

def executor_node(state):
    sql_query = state.get("sql_query", "")

    if not sql_query:
        state["sql_result"] = {
            "success": False,
            "rows": [],
            "error": "No SQL query generated"
        }
        return state

    success, rows, error = db.run_sql(sql_query)

    state["sql_result"] = {
        "success": success,
        "rows": rows,
        "error": error
    }

    return state

def planner_node(state: AgentState):
    question = state["question"]
    plan = planner(question)

    state["planner_output"] = plan
    return state

def nlsql_node(state: AgentState):

    state["sql_query"] = ""
    return state




def synthesizer_node(state: AgentState):
    """
    Produce final answer with:
      - final_answer
      - citations
      - explanation
    Must match format_hint from the spec.
    """
    # TODO: call DSPy synthesis module
    state["final_answer"] = {
        "result": None,
        "unit": "USD",
        "explanation": "",
        "citations": []
    }
    return state


def repair_node(state: AgentState):
    """
    If SQL query failed or output is invalid:
       retry NL→SQL or Synthesizer up to 2 times.
    """
    # TODO: implement retry logic
    return state



def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("planner", planner_node)
    graph.add_node("nlsql", nlsql_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("repair", repair_node)

    # Edges (flow)
    graph.set_entry_point("router")

    graph.add_edge("router", "retriever")      # rag/hybrid
    graph.add_edge("retriever", "planner")     # hybrid → plan
    graph.add_edge("planner", "nlsql")         # hybrid/sql
    graph.add_edge("nlsql", "executor")
    graph.add_edge("executor", "synthesizer")
    graph.add_edge("synthesizer", "repair")
    graph.add_edge("repair", END)

    return graph.compile()

