from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

llm_critic = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def llm_critic_eval(question: str, answer: str, sources: list):
    """
    Uses Gemini LLM to evaluate an answer against the context using criteria like
    relevance, groundedness, and fluency.

    Args:
        question (str): The user question.
        answer (str): The generated answer.
        sources (list[Document]): Retrieved context documents.

    Returns:
        dict: Dictionary with keys `relevance`, `groundedness`, `fluency`, each
              having `score` and `explanation`, or an error message if parsing fails.
    """
    context = "\n\n".join([doc.page_content for doc in sources])

    prompt = f"""
Evaluate the answer based on the question and the given context.
Give a JSON object containing relevance, groundedness, and fluency.

Question: {question}

Answer: {answer}

Context:
{context}

Return only a JSON in this format:

{{
  "relevance": {{
    "score": float (1-5),
    "explanation": str
  }},
  "groundedness": {{
    "score": float (1-5),
    "explanation": str
  }},
  "fluency": {{
    "score": float (1-5),
    "explanation": str
  }}
}}
Only return the JSON — no text before or after it.
"""

    try:
        response = llm_critic.invoke(prompt)
        content = getattr(response, "content", str(response))

        # Use regex to extract only the JSON part, stripping backticks and extra text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON block found in response.")

        json_block = match.group()
        return json.loads(json_block)

    except Exception as e:
        return {
            "error": f"Failed to evaluate with Gemini: {str(e)}",
            "raw_response": content if "content" in locals() else None,
        }
