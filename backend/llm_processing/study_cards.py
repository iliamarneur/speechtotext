"""Generation de fiches d'apprentissage depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant pedagogique expert en creation de fiches d'apprentissage.
Tu transformes du contenu audio transcrit en fiches claires, memorisables et structurees.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

STUDY_CARDS_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Genere des fiches d'apprentissage pour reviser et memoriser le contenu. Pour chaque fiche :

1. **Titre** — le concept ou sujet de la fiche
2. **Definition / Explication** — explication claire et concise (2-4 phrases)
3. **Exemple** — un exemple concret tire du contenu si possible
4. **A retenir** — la phrase cle a memoriser

Genere entre 5 et 15 fiches selon la richesse du contenu.
Separe chaque fiche par une ligne horizontale (---).
Commence par les concepts les plus fondamentaux, puis progresse vers les plus avances.

Transcription :
{text}"""


def generate_study_cards_stream(text: str, filename: str = "audio", model: str = None):
    """
    Genere des fiches d'apprentissage en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = STUDY_CARDS_PROMPT_TEMPLATE.format(filename=filename, text=text)

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
