from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict
from agent.rag.retrieval import BM25Retriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule,SynthesizerModule,PlannerModule,NLtoSQLModule
from dotenv import load_dotenv
import json
from agent.llm_set_up import load_llm_gemini
from agent.tools.repair import repair_issue
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
load_dotenv()

def generate_experiment_name():
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%y_%H-%M-%S")
    return f"Experiment_{timestamp}"

experment_name=generate_experiment_name()
import mlflow
mlflow.dspy.autolog()
mlflow.set_tracking_uri("file:///d:/Amr/ITI/Retail Analytics Copilot/mlruns")
mlflow.set_experiment(experment_name)

load_llm_gemini()
db = SQLiteTool("data/northwind.sqlite")
router = RouterModule()
retriever = BM25Retriever()
planner = PlannerModule()
nl_to_sql = NLtoSQLModule()
sql_tool = SQLiteTool()
synth = SynthesizerModule()


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



# Graph Nodes

def router_node(state: State) -> dict:
    mode = router(question=state["question"])["mode"]
    # print("Router selected mode:", mode)
    return {"mode": mode}


def retriever_node(state: State) -> dict:
    chunks = retriever.retrieve(state["question"], k=4)
    return {"rag_chunks": chunks}


def planner_node(state: State) -> dict:
    # print("Planner node activated")
    plan = planner(question=state["question"])
    return {"planner_output": plan}


def nl_to_sql_node(state: State) -> dict:
    dbschema = sql_tool.get_schema()
    past_results=state.get("sql_result", {}) if "sql_result" in state else ""
    sql = nl_to_sql(
        question=state["question"],
        planner_output=state["planner_output"],
        dbschema=dbschema,
        rag_chunks=state.get("rag_chunks", []),
        past_results=past_results,
    )['sql_query']
    return {"sql_query": sql}


def sql_node(state: State) -> dict:
    result = sql_tool.run_sql(state["sql_query"])
    return {"sql_result": result}


def synth_node(state: State) -> dict:
    out = synth.forward(
        question=state["question"],
        mode=state["mode"],
        format_hint=state.get("format_hint", ""),
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


def refactor_node(state: State) -> dict:
    retries = state.get("retries", 0)

    if retries >= 2:
        return {"route": "end"}

    next_step = repair_issue(state)

    if next_step is None:
        # print("No issues detected. Ending.")
        return {"route": "end"}
    
    # print(f"Issue detected. Routing to: {next_step}")
    return {
        "route": next_step,
        "retries": retries + 1
    }

def evaluate_node(state: State) -> dict:
    confidence = 0.5
    mode= state.get("mode", "hybrid")
    sql_result = state.get("sql_result", {})
    rag_chunks = state.get("rag_chunks", [])
    retries = state.get("retries", 0)

    confidence -= 0.05 * retries

    # SQL evaluating
    if mode in ("sql", "hybrid"):
        if not sql_result.get("success", False):
            confidence -= 0.1
        elif len(sql_result.get("rows", [])) == 0:
            confidence -= 0.1
    
    # Chunks evaluating
    if mode in ("rag", "hybrid"):
        if len(rag_chunks) == 0:
            confidence -= 0.1
        
        if len(rag_chunks) > 0:
            scores=[chunk.get("score", 0.0) for chunk in rag_chunks]
            avg_score = sum(scores)/len(scores) if scores else 0.0
            rag_contribution = min(0.2, avg_score / 20.0)
            confidence += rag_contribution

    final_confidence = max(0.01, min(1.0, confidence))
    return {"confidence": final_confidence}

def end_node(state: State) -> dict:
    print("\nFINAL OUTPUT:\n", "id: ",state['id'], "\n", "final_answer: ", state['final_answer'],"\n", "explanation: ", state.get('explanation',""), "\n", "citations: ", state.get('citations',[]), "\n", "sql: ", state.get('sql_query',""), "\n", "confidence: ", state.get('confidence',0.0))
    return {}



# Routing Logic


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



    

# Build Graph


graph = StateGraph(State)

# nodes
graph.add_node("router", router_node)
graph.add_node("retriever", retriever_node)
graph.add_node("planner", planner_node)
graph.add_node("nl2sql", nl_to_sql_node)
graph.add_node("execute_sql", sql_node)
graph.add_node("synth", synth_node)
graph.add_node("evaluator", evaluate_node)
graph.add_node("refactor", refactor_node)
graph.add_node("end", end_node)

graph.set_entry_point("router")

# --- ROUTERING ---
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
        "rag_end": "synth", # RAG mode => to end
        "go_planner": "planner", # SQL mode (normally) and HYBIRD
    }
)


# --- SQL & HYBRID common route ---
graph.add_edge("planner", "nl2sql")
graph.add_edge("nl2sql", "execute_sql")
graph.add_edge("execute_sql", "synth")

graph.add_edge("synth", "evaluator") 
graph.add_edge("evaluator", "refactor") 


# --- REPAIRING LOOP ---
graph.add_conditional_edges(
    "refactor",
    lambda s: s["route"],
    {
        "nl2sql": "nl2sql",
        "synth": "synth",
        "end": "end"
    }
)

# --- ENDING ---
graph.add_edge("end", END)

# app = graph.compile(checkpointer=memory)
# print(app.get_graph().draw_mermaid())
app = graph.compile()