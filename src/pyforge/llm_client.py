"""Request coaching feedback and reference explanations from Ollama.

            The model is selected through OLLAMA_MODEL, defaulting to llama3.1:8b.
            Environment variables are loaded from .env.

"""
from ollama import chat
from ollama import ChatResponse
from dotenv import load_dotenv
import os

load_dotenv()


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

    """Generate a reference explanation independently of a student's answer.

            Args:
                concept: The Python topic or explanation question to address.

            Returns:
                A model-generated explanation requested in three sentences,
                without code examples or questions.
    """
    system_prompt = "Explain a Python topic accurately and simply, in three sentences, without code examples or questions."
    user_prompt   = concept
    
    return ask_model(system_prompt=system_prompt, user_prompt=user_prompt)

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
