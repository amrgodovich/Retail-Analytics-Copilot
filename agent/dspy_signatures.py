import dspy
from dspy import InputField, OutputField, Predict,ChainOfThought
import json


class Routersignature(dspy.Signature):
    question: str = InputField(desc="the user question")
    mode: str = OutputField(desc="to answer this question, should we use rag using documents of the company, sql query from our database, or hybrid, answer with one choice/word from tese ['rag','sql','hybrid']")

class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(Routersignature)

    def forward(self, question: str):
        result = self.predict(question=question)
        mode = str(result.mode).strip().lower()
    
        return {"mode": mode}
        # return result


class PlannerSignature(dspy.Signature):
    """ -date ranges
        - category
        - KPI
    """
    question = InputField(desc="the user question")

    start_date = OutputField(desc="start date extracted from question, YYYY-MM-DD or text")
    end_date = OutputField(desc="end date extracted from question, YYYY-MM-DD or text")
    category = OutputField(desc="product category or item, e.g., Beverages, Produce, Dairy")
    kpi = OutputField(desc="sales, revenue, orders, margin, etc.")
    filters = OutputField(desc="any additional filters like customer name, region, employee")


class PlannerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(PlannerSignature)

    def forward(self, question: str):
        result = self.predict(question=question)
        return {
            "start_date": result.start_date,
            "end_date": result.end_date,
            "category": result.category,
            "kpi": result.kpi,
            "filters": result.filters
        }



import dspy
import json
from dspy import InputField, OutputField, Predict, ChainOfThought

class SynthesizerSignature(dspy.Signature):
    """You are a Retail Analytics Synthesizer.
    
    Your task:
    - Combine SQL results and retrieved documents to answer the user's question
    - The final_answer MUST match the format_hint type exactly (int, float, object, list, etc.)
    - Include citations from:
      * DB tables used: Orders, Order Details, Products, Customers, Categories, Suppliers
      * Doc chunks: e.g., marketing_calendar::chunk0, kpi_definitions::chunk1
    - Return valid JSON with keys: final_answer, citations, explanation
    """
    
    question= InputField(desc="The user's analytics question")
    mode= InputField(desc="rag, sql, or hybrid")
    format_hint= InputField(desc="Required output type: int, float, object, list, etc.")
    planner_output=  InputField(desc="Extracted constraints: date ranges, categories, KPIs")
    rag_chunks= InputField(desc="Retrieved doc chunks: [{id, text, source}, ...]")
    sql_result=  InputField(desc="SQL execution result: {success, rows, error, sql}")
    
    answer_json= OutputField(desc="JSON string with final_answer (matching format_hint), citations, explanation")


class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(SynthesizerSignature)

    def forward(self, question, mode, format_hint, planner_output, rag_chunks, sql_result):
        result = self.predict(
            question=question,
            mode=mode,
            format_hint=format_hint,
            planner_output=planner_output,
            rag_chunks=rag_chunks,
            sql_result=sql_result,
        )

        print("Synthesizer raw output:", result)
        # JSON output
        try:
            answer_dict = json.loads(result.answer_json)
        except json.JSONDecodeError:
            answer_dict = {
                "final_answer": None,
                "citations": [],
                "explanation": "Failed to parse response"
            }
        try:
            final = answer_dict.get("final_answer")
            
            if format_hint == "int":
                final = int(final)
            elif format_hint == "float":
                final = float(final)
            
        except (ValueError, TypeError):
            answer_dict["final_answer"] = None

        return answer_dict


class NLtoSQLSignature(dspy.Signature):
    question = InputField(desc="the user question")
    planner_output = InputField(desc="the extracted constraints from the planner")
    dbschema = InputField(desc="database schema as a string")
    sql_query = OutputField(desc="the generated SQL query to run against the database")

class NLtoSQLModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict(NLtoSQLSignature)

    def forward(self, question: str, planner_output: dict, dbschema: str):
        result = self.predict(question=question, planner_output=planner_output, dbschema=dbschema)
        sql_query = str(result.sql_query).strip() if getattr(result, "sql_query", None) is not None else ""
        return {"sql_query": sql_query}

