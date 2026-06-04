
from pathlib import Path
import json
import sys


def find_project_root(start=None, marker=("data", "lai2023")):
    """
    Walk upward from `start` until a directory containing `marker` is found.

    Default marker checks for:
        <root>/data/lai2023
    """
    start = Path.cwd() if start is None else Path(start).resolve()

    for path in [start, *start.parents]:
        marker_path = path.joinpath(*marker)
        if marker_path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find project root containing {'/'.join(marker)} from {start}"
    )


def add_project_root_to_syspath(project_root):
    """
    Useful in notebooks if imports from src.analysis fail.
    """
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
