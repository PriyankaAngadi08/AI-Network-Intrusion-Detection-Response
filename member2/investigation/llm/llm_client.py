
import json
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def investigate_with_llm(prompt: str) -> dict:
    """Generate an AI investigation using a local Ollama model."""

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running."
        ) from exc

    content = data.get("response", "")

    if not content:
        raise RuntimeError("Ollama returned an empty response")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Ollama did not return valid JSON") from exc

    if not isinstance(result, dict):
        raise ValueError("The LLM response must be a JSON object")

    return result
