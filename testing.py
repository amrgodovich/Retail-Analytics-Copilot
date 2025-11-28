# from ollama import Client

# client = Client()

# response = client.chat(
#     model="phi3.5",
#     messages=[{"role": "user", "content": "Hello, how are you?"}]
# )

# print(response["message"]["content"])

# from agent.llm_setup import load_llm
from agent.dspy_signatures import RouterModule

import dspy

def load_llm():
    """
    Configure DSPy to use your local Ollama model.
    Must be called before using any DSPy modules.
    """
    llm = dspy.LM("ollama/phi3.5")
    dspy.configure(lm=llm)
    return llm

load_llm() 

router = RouterModule()
print(router("What were sales in March 2024?"))
