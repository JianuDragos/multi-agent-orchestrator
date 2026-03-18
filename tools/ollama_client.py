import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class OllamaClient:
    def __init__(self, config_or_path="config.json"):
        if isinstance(config_or_path, dict):
            self.config = config_or_path
            self.config_path = None
        else:
            raw_path = Path(config_or_path)
            if raw_path.is_absolute():
                self.config_path = raw_path
            else:
                repo_root = Path(__file__).resolve().parent.parent
                self.config_path = (repo_root / raw_path).resolve()

            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        ollama_cfg = self.config.get("ollama", {})
        self.base_url = str(ollama_cfg.get("base_url", "http://127.0.0.1:11434/api")).rstrip("/")
        self.timeout_sec = int(ollama_cfg.get("timeout_sec", 300))
        self.max_retries = int(ollama_cfg.get("max_retries", 1))
        self.base_backoff_sec = int(ollama_cfg.get("base_backoff_sec", 1))
        self.keep_alive_default = ollama_cfg.get("keep_alive_default", "5m")
        self.keep_alive_by_role = dict(ollama_cfg.get("keep_alive_by_role", {}))
        self.options_default = dict(ollama_cfg.get("options", {}))
        self.think = ollama_cfg.get("think", False)
        self.system_prompt_by_role = dict(ollama_cfg.get("system_prompt_by_role", {}))

    def get_model(self, role: str) -> str:
        models = self.config.get("models", {})
        if role in models:
            return models[role]
        if "default" in models:
            return models["default"]
        raise KeyError(f"Missing model mapping for role: {role}")

    def get_keep_alive(self, role: str):
        return self.keep_alive_by_role.get(role, self.keep_alive_default)

    def get_system_prompt(self, role: str, override: str = "") -> str:
        if override:
            return override
        return str(self.system_prompt_by_role.get(role, "")).strip()

    def _post_json(self, endpoint: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from Ollama API:\n{raw}") from exc

    def generate(self, role: str, prompt: str, system_prompt: str = "") -> str:
        model = self.get_model(role)
        keep_alive = self.get_keep_alive(role)
        system = self.get_system_prompt(role, system_prompt)
        last_error = None

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "think": self.think,
            "keep_alive": keep_alive,
            "options": self.options_default,
        }

        for attempt in range(1, self.max_retries + 1):
            print(f"[OllamaAPI] role={role} model={model} attempt={attempt}/{self.max_retries}")
            try:
                data = self._post_json("/generate", payload)

                if not isinstance(data, dict):
                    last_error = f"Expected dict from Ollama, got {type(data).__name__}"
                else:
                    text = str(data.get("response", "")).strip()
                    if text:
                        return text
                    last_error = (
                        f"Empty Ollama response for role '{role}' with model '{model}'. "
                        f"Request succeeded but response was empty."
                    )

            except urllib.error.HTTPError as exc:
                with exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                last_error = (
                    f"Ollama HTTP error for role '{role}' with model '{model}' "
                    f"(attempt {attempt}/{self.max_retries}): {exc.code}\n{detail}"
                )
            except urllib.error.URLError as exc:
                last_error = (
                    f"Ollama connection error for role '{role}' with model '{model}' "
                    f"(attempt {attempt}/{self.max_retries}): {exc}"
                )
            except Exception as exc:
                last_error = (
                    f"Unexpected Ollama API failure for role '{role}' with model '{model}' "
                    f"(attempt {attempt}/{self.max_retries}): {exc}"
                )

            if attempt < self.max_retries:
                time.sleep(self.base_backoff_sec * (2 ** (attempt - 1)))

        raise RuntimeError(last_error or f"Unknown Ollama API failure for role '{role}'")
