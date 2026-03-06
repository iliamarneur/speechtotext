"""Generation d'infographie (spec Vega-Lite JSON) depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en visualisation de donnees.
Tu extrais des donnees quantitatives ou categoriques d'un contenu audio et tu generes des specifications Vega-Lite valides.
Tu rediges les labels en francais."""

INFOGRAPHIC_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Analyse le contenu et genere UNE specification Vega-Lite (JSON valide) qui visualise les donnees ou concepts cles.

Regles strictes :
- Le JSON doit etre une spec Vega-Lite valide (schema version 5)
- Utilise "$schema": "https://vega.github.io/schema/vega-lite/v5.json"
- Choisis le type de chart le plus adapte : bar, pie (arc), line, point, etc.
- Les donnees doivent etre incluses inline dans "data.values"
- Ajoute un "title" descriptif en francais
- Si le contenu est narratif sans donnees chifrees, cree un bar chart des themes principaux avec leur importance relative (1-10)
- width: 500, height: 300
- Utilise "color" pour distinguer les categories

Genere UNIQUEMENT le JSON Vega-Lite, sans aucun texte avant ou apres. Pas de ```json```, juste le JSON pur.

Transcription :
{text}"""


def generate_infographic_stream(text: str, filename: str = "audio", model: str = None, custom_instructions: str = None):
    """
    Genere une infographie en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = INFOGRAPHIC_PROMPT_TEMPLATE.format(filename=filename, text=text)
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
