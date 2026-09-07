from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def pick_llm(level:str):
    """
    picks the appropriate llm based on the level of the question

    Args:
    level (str):the level of the question can b "easy","medium","high"

    
    """

    if level.lower()=="low":
        llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0)
    elif level.lower()=="medium":
        llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0)
    elif level.lower()=="high":
        llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0)
    else:
        raise ValueError(f"Unsupported level : {level}")

    return llm
        