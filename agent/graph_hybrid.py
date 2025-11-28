from agent.llm_setup import load_llm
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, Optional
from agent.rag.retrieval import BM25Retriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule,SynthesizerModule,PlannerModule,NLtoSQLModule



load_llm()   
router = RouterModule()
planner = PlannerModule()
synthesizer = SynthesizerModule()
db = SQLiteTool("data/northwind.sqlite")
retriever = BM25Retriever(docs_folder="docs")
nl_to_sql= NLtoSQLModule()

class AgentState(dict):
    id: str
    question: str
    format_hint: str
    mode: str
    planner_output: dict
    rag_chunks: list
    sql_query: str
    sql_result: dict
    final_answer: dict
    retries: int
    confidence: float

def calculate_confidence(state):
    """
    Calculate confidence score based on mode and data quality.
    - RAG mode: checks retrieval quality + chunk scores
    - SQL mode: checks query success + non-empty rows
    - Hybrid mode: combines both
    - Penalty for repairs
    """
    mode = state.get("mode", "hybrid")
    confidence = 0.5
    
    # rag
    if mode == "rag":
        rag_chunks = state.get("rag_chunks", [])
        
        if not rag_chunks:
            confidence = 0.2  # No chunks
        else:
            # Check chunk quality
            scores = [chunk.get("score", 0.5) for chunk in rag_chunks]
            avg_score = sum(scores) / len(scores)
            
            # High if avg > 0.7
            if avg_score > 0.7:
                confidence = 0.85
            elif avg_score > 0.5:
                confidence = 0.7
            else:
                confidence = 0.4
            
            if len(rag_chunks) >= 3:
                confidence = confidence+0.1
    
    # sql
    elif mode == "sql":
        sql_result = state.get("sql_result", {})
        
        if not sql_result.get("success"):
            confidence = 0.1  # query failed
        elif len(sql_result.get("rows", [])) == 0:
            confidence = 0.3  # Query succeeded but no rows
        else:
            rows = sql_result.get("rows", [])
            if len(rows) >= 5:
                confidence = 0.9  
            elif len(rows) >= 1:
                confidence = 0.8
            else:
                confidence = 0.5
    
    # hybrid
    elif mode == "hybrid":
        rag_chunks = state.get("rag_chunks", [])
        sql_result = state.get("sql_result", {})
        
        rag_quality = 0.0
        sql_quality = 0.0
        
        # RAG
        if rag_chunks:
            scores = [chunk.get("score", 0.0) for chunk in rag_chunks]
            avg_score = sum(scores) / len(scores)
            rag_quality = avg_score * 0.5  # Max 0.5
        
        # SQL 
        if sql_result.get("success") and len(sql_result.get("rows", [])) > 0:
            sql_quality = 0.5  # Max 0.5
        
        confidence = rag_quality + sql_quality
        if confidence == 0:
            confidence = 0.3  # Fallback
    
    # minus for retries
    retries = state.get("retries", 0)
    if retries > 0:
        confidence *= (1 - 0.15 * retries)  # -15% per repair
    
    return round(min(max(confidence, 0.0), 1.0), 2)


def router_node(state: AgentState):
    """
        rag | sql | hybrid 
    """
    print("first router node called")
    question = state["question"]
    result = router(question)
    state["mode"] = result["mode"]
    print("Router selected mode:", state["mode"])
    if state["mode"] not in ["rag", "sql", "hybrid"]:
            state["mode"] = "hybrid"
    return state

def retriever_node(state: AgentState):
    mode = state.get("mode", "rag")

    if mode not in ["rag", "hybrid"]:
        state["rag_chunks"] = []
        return state
    print("retriever node called")
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
            "error": "No SQL query generated",
            "sql": ""
        }
        return state
    print("executor node called")
    state["sql_result"] = db.run_sql(sql_query)
    print("Executed res: ", state["sql_result"])
    return state

def planner_node(state: AgentState):
    question = state["question"]
    print("planner node called")
    plan = planner(question)
    print("Planner output:", plan)
    state["planner_output"] = plan
    return state

def synthesizer_node(state: AgentState):
    print(state)
    print("synthesizer node called") 
    synth_result = synthesizer(
        question=state["question"],
        mode=state.get("mode", "hybrid"),
        format_hint=state.get("format_hint", "object"),
        planner_output=state.get("planner_output", {}),
        rag_chunks=state.get("rag_chunks", []),
        sql_result=state.get("sql_result", {}),
    )

    sql_query = state.get("sql_result", {}).get("sql", "")
    
    state["final_answer"] = {
        "id": state.get("id", ""),
        "final_answer": synth_result.get("final_answer"),
        "sql": sql_query,
        "confidence": calculate_confidence(state),
        "explanation": synth_result.get("explanation", ""),
        "citations": synth_result.get("citations", []),
    }
    
    return state


def nlsql_node(state: AgentState):
    """Generate SQL query from question using NL→SQL module."""
    mode = state.get("mode", "rag")
    
    if mode == "rag":
        state["sql_query"] = ""
        return state
    
    question = state["question"]
    planner_output = state.get("planner_output", {})
    print("nlsql node called")
    schema_str = str(db.get_schema())
    print("Database schema:", schema_str)
    result = nl_to_sql(question, planner_output, schema_str)
    state["sql_query"] = result.get("sql_query", "")
    print("Generated SQL query:", state["sql_query"])

    
    return state


def repair_node(state: AgentState):
    """
    Retry SQL generation + execution if:
      - SQL error
      - Empty SQL query
      - Rows empty in SQL mode
    Retries up to 2 times.
    """
    mode = state.get("mode", "hybrid")
    retries = state.get("retries", 0)
    print("repair node called, current retries:", retries)

    if mode == "rag":
        return state

    if retries >= 2:
        return state

    sql_result = state.get("sql_result", {})
    sql_success = sql_result.get("success", False)
    sql_query = sql_result.get("sql", "")
    rows = sql_result.get("rows", [])

    needs_repair = False

    if not sql_query:
        needs_repair = True
    elif not sql_success:
        needs_repair = True
    elif mode == "sql" and len(rows) == 0:
        needs_repair = True

    if not needs_repair:
        return state

    retries += 1
    state["retries"] = retries

    question = state["question"]
    planner_output = state.get("planner_output", {})
    schema_str = str(db.get_schema())

    new_sql = nl_to_sql(question, planner_output, schema_str)
    new_query = new_sql.get("sql_query", "")

    state["sql_query"] = new_query

    state["sql_result"] = db.run_sql(new_query)

    print("Repair attempt", retries, "generated new SQL:", new_query)

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

    graph.add_edge("router", "retriever")      
    graph.add_edge("retriever", "planner")     
    graph.add_edge("planner", "nlsql")
    graph.add_edge("nlsql", "executor")
    graph.add_edge("executor", "synthesizer")
    graph.add_edge("synthesizer", "repair")
    graph.add_edge("repair", END)

    return graph.compile()

