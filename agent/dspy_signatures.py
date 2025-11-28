import dspy

class Routersignature(dspy.Signature):
    question: str = dspy.InputField(desc="the user question")
    mode: str = dspy.OutputField(desc="to answer this question, should we use rag using documents of the company, sql query from our database, or hybrid, answer with one choice/word from tese ['rag','sql','hybrid']")

class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Routersignature)

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
    question = dspy.InputField("user question")

    start_date = dspy.OutputField(desc="start date extracted from question, YYYY-MM-DD or text")
    end_date = dspy.OutputField(desc="end date extracted from question, YYYY-MM-DD or text")
    category = dspy.OutputField(desc="product category or item, e.g., Beverages, Produce, Dairy")
    kpi = dspy.OutputField(desc="sales, revenue, orders, margin, etc.")
    filters = dspy.OutputField(desc="any additional filters like customer name, region, employee")


class PlannerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PlannerSignature)

    def forward(self, question: str):
        result = self.predict(question=question)
        return {
            "start_date": result.start_date,
            "end_date": result.end_date,
            "category": result.category,
            "kpi": result.kpi,
            "filters": result.filters
        }