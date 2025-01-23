# services/openai.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Correct the variable assignment
# model = 'gpt-4o-mini-realtime-preview'
model = 'chatgpt-4o-latest'

def generate_response(prompt, model=model, max_tokens=150): # DONT CHANGE THIS MODEL
    """
    Generates a response from OpenAI's GPT model based on the given prompt.

    Args:
        prompt (str): The input text prompt.
        model (str): The OpenAI model to use.
        max_tokens (int): The maximum number of tokens in the response.

    Returns:
        str: The generated text response.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error communicating with OpenAI API: {str(e)}")
        return "Sorry, I couldn't process your request at the moment."
