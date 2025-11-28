import dspy

class Routersignature(dspy.Signature):
    question: str = dspy.InputField(desc="the user question")
    mode: str = dspy.OutputField(desc="rag, sql, or hybrid")



class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Routersignature)

    def forward(self, question: str):
        result = self.predict(question=question)
        mode = str(result.mode).strip().lower()
    
        return {"mode": mode}