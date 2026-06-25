"""
LLM Client for Memora.
A reusable wrapper around the Qwen Cloud API.
Other modules call these functions instead of setting up the API each time.
"""

from openai import OpenAI
from app.config import settings

# Create ONE client object, reused for every call.
# It reads the key and URL from our central config.
client = OpenAI(
    api_key=settings.QWEN_API_KEY,
    base_url=settings.QWEN_BASE_URL,
)


def chat(messages, model=None, temperature=0.7):
    """
    Send a conversation to Qwen and get the model's reply.

    INPUT:
      - messages: a list of message dicts, e.g.
          [{"role": "user", "content": "Hello"}]
      - model: which model to use (defaults to qwen-plus-latest from config)
      - temperature: creativity level (0 = strict/factual, 1 = creative)

    RETURNS:
      - a string: the model's text reply
    """
    # If no model was specified, use the default from config
    if model is None:
        model = settings.LLM_MODEL

    # Send the request to Qwen
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    # Extract just the text reply from the response object and return it
    return response.choices[0].message.content


def get_embedding(text):
    """
    Convert a piece of text into an embedding (a list of numbers/vector).
    This vector represents the *meaning* of the text, so similar meanings
    produce similar vectors. This is the foundation of memory search.

    INPUT:
      - text: a string, e.g. "I love pizza"

    RETURNS:
      - a list of floats (the vector), e.g. [0.013, -0.221, 0.087, ...]
    """
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )

    # The vector lives inside the response; pull it out and return it
    return response.data[0].embedding