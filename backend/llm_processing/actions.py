"""Extraction d'actions depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en analyse de contenu audio.
Tu identifies et extrais les actions, taches et decisions mentionnees dans le contenu.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

ACTIONS_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Identifie et extrais toutes les actions, taches, decisions et engagements mentionnes.
Organise-les ainsi :

1. **Actions a realiser** — liste des taches concretes a effectuer
   - Pour chaque action, precise si possible : qui, quoi, quand/delai
2. **Decisions prises** — les decisions actees durant l'echange
3. **Questions en suspens** — les points non resolus qui necessitent un suivi

Utilise des cases a cocher Markdown (- [ ]) pour les actions.
Si aucune action ou decision n'est identifiable, indique-le clairement.

Transcription :
{text}"""


def extract_actions_stream(text: str, filename: str = "audio", model: str = None, custom_instructions: str = None):
    """
    Extrait les actions en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = ACTIONS_PROMPT_TEMPLATE.format(filename=filename, text=text)
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
