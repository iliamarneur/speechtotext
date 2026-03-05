"""Résumé automatique de transcriptions via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en synthèse de contenu audio.
Tu produis des résumés clairs, structurés et fidèles au contenu original.
Tu rédiges toujours en français sauf indication contraire.
Tu utilises le format Markdown."""

SUMMARY_PROMPT_TEMPLATE = """Voici la transcription complète d'un audio intitulé "{filename}".

Génère un résumé structuré avec :
1. **Résumé** — un résumé concis en 2-3 paragraphes
2. **Points clés** — les informations essentielles sous forme de liste à puces
3. **Conclusion** — les conclusions ou messages principaux

Transcription :
{text}"""


def summarize_stream(text: str, filename: str = "audio", model: str = None):
    """
    Génère un résumé en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = SUMMARY_PROMPT_TEMPLATE.format(filename=filename, text=text)

    full_response = ""
    for chunk in generate_stream(prompt, model, system=SYSTEM_PROMPT):
        token = chunk.get("response", "")
        done = chunk.get("done", False)
        full_response += token

        result = {"token": token, "done": done}

        if done:
            # Stats du dernier chunk Ollama
            result["stats"] = {
                "total_duration_ms": chunk.get("total_duration", 0) // 1_000_000,
                "eval_count": chunk.get("eval_count", 0),
                "model": model,
            }
            result["full_text"] = full_response

        yield result
