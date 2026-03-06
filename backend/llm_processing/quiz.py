"""Generation de quiz automatique depuis une transcription via LLM local."""

from backend.config import LLM_MODEL
from backend.llm_processing.ollama_client import generate_stream

SYSTEM_PROMPT = """Tu es un assistant pedagogique expert en creation de quiz.
Tu generes des questions pertinentes et variees pour tester la comprehension d'un contenu audio.
Tu rediges toujours en francais sauf indication contraire.
Tu utilises le format Markdown."""

QUIZ_PROMPT_TEMPLATE = """Voici la transcription complete d'un audio intitule "{filename}".

Genere un quiz de revision avec 10 questions variees pour tester la comprehension du contenu.
Pour chaque question :

1. **Question N** — la question claire et precise
2. **Type** — QCM, Vrai/Faux, ou Question ouverte
3. Pour les QCM : propose 4 choix (A, B, C, D) dont une seule bonne reponse
4. Pour les Vrai/Faux : une affirmation a evaluer
5. Pour les questions ouvertes : une question demandant une reponse courte

Apres toutes les questions, ajoute une section :
## Reponses
Avec la reponse correcte et une breve explication pour chaque question.

Varie les types de questions et couvre l'ensemble du contenu.

Transcription :
{text}"""


def generate_quiz_stream(text: str, filename: str = "audio", model: str = None, custom_instructions: str = None):
    """
    Genere un quiz en streaming.

    Yields:
        dict avec 'token' (str), 'done' (bool), et optionnellement 'stats' (dict).
    """
    model = model or LLM_MODEL
    prompt = QUIZ_PROMPT_TEMPLATE.format(filename=filename, text=text)
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
