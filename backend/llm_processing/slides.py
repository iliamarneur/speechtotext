"""Generation de slides depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en creation de presentations.
Tu generes des slides claires, visuelles et synthetiques a partir de contenu audio.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

SLIDES_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Genere une presentation sous forme de slides en Markdown. Regles strictes :

- Chaque slide est separee par --- (trois tirets sur une ligne seule)
- La premiere slide est la page de titre avec # Titre et un sous-titre
- Chaque slide suivante a un titre ## et du contenu concis
- Utilise des listes a puces courtes (3-5 points par slide max)
- Utilise **gras** pour les mots cles importants
- Entre 6 et 12 slides selon la richesse du contenu
- La derniere slide est un resume / conclusion

Exemple de format :
# Titre de la presentation
Sous-titre ou contexte

---

## Slide 2 : Premier theme
- Point cle 1
- Point cle 2
- **Element important**

---

## Slide 3 : Deuxieme theme
- Detail A
- Detail B

Ne genere RIEN d'autre que les slides Markdown. Pas d'introduction ni de commentaire.

Transcription :
{text}"""


def generate_slides_stream(text: str, filename: str = "audio", model: str = None):
    """
    Genere des slides en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = SLIDES_PROMPT_TEMPLATE.format(filename=filename, text=text)

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
