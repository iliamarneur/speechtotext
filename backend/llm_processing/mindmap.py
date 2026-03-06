"""Generation de carte mentale (Markdown hierarchique) depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en structuration visuelle de contenu.
Tu generes des cartes mentales sous forme de Markdown hierarchique.
Tu rediges toujours en francais sauf indication contraire."""

MINDMAP_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Genere une carte mentale hierarchique en Markdown. Regles strictes :

- Le titre principal est un heading de niveau 1 (#)
- Les branches principales sont des headings de niveau 2 (##)
- Les sous-branches sont des headings de niveau 3 (###)
- Les feuilles sont des listes a puces (-)
- Chaque element doit etre court (5-10 mots max)
- Couvre l'ensemble du contenu de maniere structuree
- Entre 4 et 8 branches principales

Exemple de format :
# Titre du sujet
## Branche 1
### Sous-branche 1.1
- Detail A
- Detail B
### Sous-branche 1.2
- Detail C
## Branche 2
- Detail D
- Detail E

Ne genere RIEN d'autre que le Markdown hierarchique. Pas d'introduction ni de conclusion.

Transcription :
{text}"""


def generate_mindmap_stream(text: str, filename: str = "audio", model: str = None, custom_instructions: str = None):
    """
    Genere une carte mentale en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = MINDMAP_PROMPT_TEMPLATE.format(filename=filename, text=text)
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
