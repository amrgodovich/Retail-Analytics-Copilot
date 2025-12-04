# from ollama import Client

# client = Client()

# response = client.chat(
#     model="phi3.5",
#     messages=[{"role": "user", "content": "Hello, how are you?"}]
# )

# print(response["message"]["content"])

# from agent.llm_setup import load_llm
# from agent.dspy_signatures import RouterModule,PlannerModule

# import dspy

# def load_llm():
#     """
#     Configure DSPy to use your local Ollama model.
#     Must be called before using any DSPy modules.
#     """
#     llm = dspy.LM("ollama/phi3.5")
#     dspy.configure(lm=llm)
#     return llm

# load_llm() 

# # router = RouterModule()
# # print(router("What were sales in March 2024?"))

# planner = PlannerModule()
# print(planner("What were sales between March and June 2024 for Beverages?"))


# g = build_graph()
# state = {"question":"Top 3 products by revenue all-time", "format_hint":"list[{product:str, revenue:float}]", "id":"test1", "retries":0}
# res = g.invoke(state)
# print(res)
# print("Final answer:", res["final_answer"])
# from agent.graph_hybrid import run_agent_hybrid

# from agent.rag.retrieval import BM25Retriever
# from agent.graph_hybrid import build_graph, AgentState


# def run_agent_hybrid(question: str, format_hint: str = "object"):
#     """
#     Runs the full hybrid agent:
#         router → retriever → planner → nlsql → executor → synthesizer → repair.
#     Returns the final structured answer.
#     """
#     graph = build_graph()

#     # Initial empty state
#     init_state = {
#         "id": "user_request_1",
#         "question": question,
#         "format_hint": format_hint,
#         "mode": "hybrid",
#         "planner_output": {},
#         "rag_chunks": [],
#         "sql_query": "",
#         "sql_result": {},
#         "final_answer": {},
#         "retries": 0,
#         "confidence": 0.0,
#     }

#     # Run
#     final_state = graph.invoke(init_state)

#     return final_state.get("final_answer", {})

# # your test input
# item = {
#     "id": "rag_policy_beverages_return_days",
#     "question": "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.",
#     "format_hint": "int"
# }

# # run agent
# result = run_agent_hybrid(
#     question=item["question"],
#     format_hint=item["format_hint"]
# )

# # print nicely
# print("\n=== RESULT ===")
# print(result)

from agent.graph_hybrid import app
if __name__ == "__main__":
    result = app.invoke({"id":"rag_example_1","question":"According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.","format_hint":"int"})

    final_output = {
        "id": result.get("id"),
        "final_answer": result.get("final_answer"),
        "sql": result.get("sql_query",""),
        "confidence": result.get("confidence", 0.0),
        "explanation": result.get("explanation"),
        "citation": result.get("citations", [])
    }

    print(final_output)
