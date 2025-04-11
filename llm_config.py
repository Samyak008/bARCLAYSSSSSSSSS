from dotenv import load_dotenv
from crewai import Agent, Task, Crew, CrewAgentExecutor
# from crewai.llms.litellm import LiteLLM
import os
from crewai import LLM

# Load environment variables
load_dotenv()

def get_groq_llm():
    """Configure and return a Groq LLM for CrewAI"""
    
    # Check if we have a Groq API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # Configure the LLM with Groq's llama model
    llm = LLM(
        model="groq/llama3-70b-8192",  # Using llama3-70b as the default model
        api_key=groq_api_key,
        temperature=0.7,
    )
    
    print("Groq LLM configured successfully!")
    return llm