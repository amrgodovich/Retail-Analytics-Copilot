from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict
from agent.rag.retrieval import BM25Retriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule,SynthesizerModule,PlannerModule,NLtoSQLModule
from dotenv import load_dotenv
import json
from agent.llm_set_up import load_llm_gemini
load_dotenv()


load_llm_gemini()
db = SQLiteTool("data/northwind.sqlite")
router = RouterModule()
retriever = BM25Retriever()
planner = PlannerModule()
nl_to_sql = NLtoSQLModule()
sql_tool = SQLiteTool()
synth = SynthesizerModule()


# def repair_node(state: State):
#     """
#     Retry SQL generation + execution if:
#       - SQL error
#       - Empty SQL query
#       - Rows empty in SQL mode
#     Retries up to 2 times.
#     """
#     mode = state.get("mode", "hybrid")
#     retries = state.get("retries", 0)
#     print("repair node called, current retries:", retries)

#     if mode == "rag":
#         return state

#     if retries >= 2:
#         return state

#     sql_result = state.get("sql_result", {})
#     sql_success = sql_result.get("success", False)
#     sql_query = sql_result.get("sql", "")
#     rows = sql_result.get("rows", [])

#     needs_repair = False

#     if not sql_query:
#         needs_repair = True
#     elif not sql_success:
#         needs_repair = True
#     elif mode == "sql" and len(rows) == 0:
#         needs_repair = True

#     if not needs_repair:
#         return state

#     retries += 1
#     state["retries"] = retries

#     question = state["question"]
#     planner_output = state.get("planner_output", {})
#     schema_str = str(db.get_schema())

#     new_sql = nl_to_sql(question, planner_output, schema_str)
#     new_query = new_sql.get("sql_query", "")

#     state["sql_query"] = new_query

#     state["sql_result"] = db.run_sql(new_query)

#     print("Repair attempt", retries, "generated new SQL:", new_query)

    # return state
# ------------------------------------------------------------
# 1. Define State
# ------------------------------------------------------------
class State(TypedDict, total=False):
    id: str
    question: str
    format_hint: str
    mode: str
    rag_chunks: list
    planner_output: dict
    sql_query: str
    sql_result: dict
    final_answer: Any
    explanation: str
    citations: list
    confidence: float
    retries: int


# ------------------------------------------------------------
# 2. Graph Nodes
# ------------------------------------------------------------

def router_node(state: State) -> dict:
    mode = router(question=state["question"])["mode"]
    print("Router selected mode:", mode)
    return {"mode": mode}


def retriever_node(state: State) -> dict:
    chunks = retriever.retrieve(state["question"], k=4)
    return {"rag_chunks": chunks}


def planner_node(state: State) -> dict:
    print("Planner node activated")
    plan = planner(question=state["question"])
    return {"planner_output": plan}


def nl_to_sql_node(state: State) -> dict:
    dbschema = sql_tool.get_schema()
    sql = nl_to_sql(
        question=state["question"],
        planner_output=state["planner_output"],
        dbschema=dbschema,
        rag_chunks=state.get("rag_chunks", []),
    )['sql_query']
    return {"sql_query": sql}


def sql_node(state: State) -> dict:
    result = sql_tool.run_sql(state["sql_query"])
    return {"sql_result": result}


def synth_node(state: State) -> dict:
    out = synth.forward(
        question=state["question"],
        mode=state["mode"],
        format_hint=state.get("format_hint", "object"),
        planner_output=state.get("planner_output", {}),
        rag_chunks=state.get("rag_chunks", []),
        sql_result=state.get("sql_result", {"success": False, "rows": [], "sql": ""}),
    )

    json_str = out.answer_json
    if json_str.startswith("```json"):
        json_str = json_str.strip("` \n")
        json_str = json_str[4:].strip()
        json_str = json_str.strip("` \n")
    json_str = json_str[:json_str.rfind("}")+1]
    output_json = json.loads(json_str)

    return {
        "final_answer": output_json.get("final_answer", ""),
        "explanation": output_json.get("explanation", ""),
        "citations": output_json.get("citations", []),
        "confidence": float(output_json.get("confidence", 0.0))
    }


def end_node(state: State) -> dict:
    print("\nFINAL STATE:\n", state, "\n")
    return {}


# ------------------------------------------------------------
# 3. Conditional Routing Logic
# ------------------------------------------------------------

def route_after_router(state: State):
    mode = state.get("mode")
    if mode == "rag":
        return "rag"
    elif mode == "sql":
        return "sql"
    else:
        return "hybrid"

def route_after_retriever(state: State):
    mode = state.get("mode")
    if mode == "rag":
        return "rag_end"
    else:
        return "go_planner"

# ------------------------------------------------------------
# 4. Build Dynamic Graph
# ------------------------------------------------------------

graph = StateGraph(State)

# nodes
graph.add_node("router", router_node)
graph.add_node("retriever", retriever_node)
graph.add_node("planner", planner_node)
graph.add_node("nl2sql", nl_to_sql_node)
graph.add_node("execute_sql", sql_node)
graph.add_node("synth", synth_node)
graph.add_node("end", end_node)

graph.set_entry_point("router")

# --- DYNAMIC BRANCHES ---
graph.add_conditional_edges(
    "router",
    route_after_router,
    {
        "rag": "retriever",
        "hybrid": "retriever",
        "sql": "planner",
    }
)

# --- RAG OR HYBIRD ---
graph.add_conditional_edges(
    "retriever",
    route_after_retriever,
    {
        "rag_end": "synth", # RAG mode= end
        "go_planner": "planner", # SQL mode (normally) and HYBIRD
    }
)


# --- SQL & HYBRID common route ---
graph.add_edge("planner", "nl2sql")
graph.add_edge("nl2sql", "execute_sql")
graph.add_edge("execute_sql", "synth")

# --- ENDING ---
graph.add_edge("synth", "end")
graph.add_edge("end", END)

app = graph.compile()