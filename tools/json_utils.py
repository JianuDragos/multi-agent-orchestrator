import json


def strip_code_fences(raw_text: str) -> str:
    lines = []
    for line in raw_text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_json_candidates(raw_text: str):
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty model output")

    text = strip_code_fences(raw_text)
    decoder = json.JSONDecoder()
    candidates = []

    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                candidates.append({
                    "index": i,
                    "object": obj,
                })
        except json.JSONDecodeError:
            continue

    if not candidates:
        raise ValueError(f"No valid JSON object found in model output:\n{text}")

    return candidates
