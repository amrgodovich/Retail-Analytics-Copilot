import dspy
import os
from dotenv import load_dotenv

load_dotenv()
def load_llm_gemini():
    """
    Configure DSPy to use the Google Gemini API.
    """
    # 1. Get the API Key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY environment variable not set.")
        # Raise an error or exit if the key is mandatory
        # return None 

    # 2. Configure DSPy to use the Gemini model
    # gemini-2.5-flash is fast and great for a free tier/general tasks
    llm = dspy.LM(
        model="gemini/gemini-2.5-flash", 
        api_key=gemini_api_key
    )
    
    dspy.configure(lm=llm)
    print("DSPy configured with Gemini 2.5 Flash!")
    return llm

# def load_llm():
#     """
#     Configure DSPy to use your local Ollama model.
#     Must be called before using any DSPy modules.
#     """
#     llm = dspy.LM("ollama/phi3.5")
#     dspy.configure(lm=llm)
#     return llm