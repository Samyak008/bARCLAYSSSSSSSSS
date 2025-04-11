from dotenv import load_dotenv
import os
from crewai import LLM
import sys

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test if Groq API key is valid and working"""
    print("Testing Groq API connection...")
    
    # Check if we have a Groq API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return False
    
    # Try to create and use the LLM
    try:
        # Configure the LLM with Groq's model
        llm = LLM(
            model="groq/llama3-70b-8192",  # Without the provider prefix
            api_key=groq_api_key,
            temperature=0.7,
            stream=True  # Enable streaming for better experience
        )
        
        # Simple test to see if the connection works using call() instead of generate()
        print("Sending test message to Groq API...")
        response = llm.call("Hello! This is a test message to check if the Groq API is working.")
        
        print("\nGroq API test successful!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"Error testing Groq API: {e}")
        return False

if __name__ == "__main__":
    success = test_groq_connection()
    if not success:
        sys.exit(1)