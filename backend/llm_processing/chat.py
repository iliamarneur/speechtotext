"""Chat avec une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert qui repond aux questions sur le contenu d'un audio transcrit.
Tu reponds de maniere precise et concise en te basant uniquement sur le contenu de la transcription.
Si la reponse n'est pas dans la transcription, dis-le clairement.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

CHAT_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Transcription :
{text}

---

Question de l'utilisateur :
{question}"""


def chat_stream(text: str, question: str, filename: str = "audio", model: str = None):
    """
    Repond a une question sur la transcription en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = CHAT_PROMPT_TEMPLATE.format(filename=filename, text=text, question=question)

    full_response = ""
    for chunk in generate_stream(prompt, model, system=SYSTEM_PROMPT):
        token = chunk.get("response", "")
        done = chunk.get("done", False)
        full_response += token

        result = {"token": token, "done": done}

        if done:
            result["stats"] = {
                "total_duration_ms": chunk.get("total_duration", 0) // 1_000_000,
                "eval_count": chunk.get("eval_count", 0),
                "model": model,
            }
            result["full_text"] = full_response

        yield result
