"""Extraction de points cles depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en analyse de contenu audio.
Tu extrais les points cles de maniere structuree, precise et fidele au contenu original.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

KEY_POINTS_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Extrais les points cles en les organisant par theme. Pour chaque theme :
1. **Titre du theme** en gras
2. Liste a puces des points essentiels sous ce theme
3. Pour chaque point, sois precis et concis (1-2 phrases max)

Identifie entre 3 et 8 themes selon la richesse du contenu.
Termine par une section **A retenir** avec les 3 points les plus importants.

Transcription :
{text}"""


def extract_key_points_stream(text: str, filename: str = "audio", model: str = None, custom_instructions: str = None):
    """
    Extrait les points cles en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = KEY_POINTS_PROMPT_TEMPLATE.format(filename=filename, text=text)
    if custom_instructions:
        prompt += "\n\nInstructions supplementaires de l'utilisateur :\n" + custom_instructions

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
