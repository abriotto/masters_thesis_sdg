from pathlib import Path
from typing import Optional
import json

def find_repo_root(start: Optional[Path] = None, repo_name: str = "masters_thesis_sdg") -> Path:
    if start is None:
        start = Path.cwd().resolve()

    current = start
    while True:
        if current.name == repo_name:
            return current
        if current.parent == current:
            raise FileNotFoundError(f"Could not find repo root '{repo_name}' from {start}")
        current = current.parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()