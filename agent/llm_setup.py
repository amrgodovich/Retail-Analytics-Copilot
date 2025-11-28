import dspy

def load_llm():
    """
    Configure DSPy to use your local Ollama model.
    Must be called before using any DSPy modules.
    """
    llm = dspy.LM("ollama/phi3.5")
    dspy.configure(lm=llm)
    return llm
