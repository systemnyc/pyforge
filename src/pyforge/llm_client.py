"""Request coaching feedback and reference explanations from Ollama.

            The model is selected through OLLAMA_MODEL, defaulting to llama3.1:8b.
            Environment variables are loaded from .env.

"""
from ollama import chat
from ollama import ChatResponse
from dotenv import load_dotenv
import os

load_dotenv()

def create_exercise(system_prompt: str) -> str:
    """"
        Sent instructions on generating python exercises

        Args:
            system_prompt: Instructions on creating python exercises
        Returns:
            The model's prompt exercise text.
        
    """

    response: ChatResponse = chat(
        model= os.getenv("OLLAMA_MODEL", "llama3.18b"),
        messages=[
            {"role": "system", "content": system_prompt},
         
        ], 
        options = {
            "temperature": 0,
        },
    )
def ask_model(system_prompt: str, user_prompt: str) -> str:
    
    """Send system instructions and user content to the configured model.

            Args:
                system_prompt: Instructions controlling the model's response.
                user_prompt: Content the model should review or explain.

            Returns:
                The model's response text.

            Note:
                Temperature is set to zero to reduce response variability.
                Model responses still require validation; accuracy is not guaranteed.
                Connection and model errors propagate to the caller.
    """

    response: ChatResponse = chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0,
        },
    )

    return response.message.content

def generate_reference(concept: str) -> str:
    """Generate a reference explanation focused on the supplied question."""
    system_prompt = (
        "You explain Python concepts accurately in plain language. "
        "Answer the supplied explanation question directly in two or three sentences. "
        "If it refers to submitted code that you have not been given, explain "
        "the general principle without inventing variable names or values. "
        "For list or tuple indexing, explain that positive indexing starts "
        "at zero and an index selects the item at that position. "
        "Do not include code examples, questions, or Markdown formatting."
    )

    return ask_model( system_prompt=system_prompt, user_prompt=concept,
    )

def grade_submission(system: str, user: str) -> str:

    """Request submission feedback using separate system and user messages.

            Args:
                system: Instructions defining how to assess the submission.
                user: Submission content and assessment context.

            Returns:
                The model's feedback text.
    """
       
    response: ChatResponse = chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response.message.content
