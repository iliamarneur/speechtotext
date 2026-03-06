"""Extraction de tableaux de donnees depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant expert en extraction et structuration de donnees.
Tu analyses du contenu audio transcrit et tu en extrais des tableaux de donnees pertinents.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

DATA_TABLE_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Analyse le contenu et extrais les donnees sous forme de tableaux Markdown. Regles strictes :

- Identifie toutes les donnees structurables : chiffres, dates, noms, comparaisons, listes, etc.
- Cree un ou plusieurs tableaux Markdown avec des en-tetes clairs
- Chaque tableau doit avoir un titre descriptif en **gras** au-dessus
- Utilise le format Markdown standard : | Col1 | Col2 | Col3 |
- Aligne les colonnes avec --- dans la ligne separatrice
- Si le contenu contient des comparaisons, cree un tableau comparatif
- Si le contenu contient des etapes/processus, cree un tableau chronologique
- Si le contenu est narratif sans donnees evidentes, cree un tableau de synthese avec les themes principaux, sous-themes et details cles
- Ajoute une ligne de **synthese** ou **total** si pertinent
- Maximum 5 tableaux

Ne genere RIEN d'autre que les tableaux Markdown. Pas d'introduction ni de commentaire.

Transcription :
{text}"""


def extract_data_table_stream(text: str, filename: str = "audio", model: str = None, custom_instructions: str = None):
    """
    Extrait des tableaux de donnees en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = DATA_TABLE_PROMPT_TEMPLATE.format(filename=filename, text=text)
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
