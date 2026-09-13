import json
import urllib.request


def generate(prompt: str, model: str, endpoint: str = "http://localhost:11434/api/generate") -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["response"]
