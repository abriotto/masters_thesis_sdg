import json
import re
from typing import Optional


def extract_first_json_block(text: str) -> Optional[str]:
    fenced_json = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_json:
        return fenced_json.group(1)

    fenced_any = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_any:
        return fenced_any.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def parse_model_json(text: str):
    candidate = extract_first_json_block(text)
    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None