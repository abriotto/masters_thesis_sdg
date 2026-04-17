
from huggingface_hub import snapshot_download
from pathlib import Path

# 1. Route to your raw data folder
project_root = Path.cwd().parents[1]
raw_dir = project_root / "data" / "lai2023"
raw_dir.mkdir(parents=True, exist_ok=True)

# 2. Download ONLY the text/JSON files, ignoring videos
print("Downloading raw files (excluding videos)...")
snapshot_download(
    repo_id="bolinlai/Werewolf-Among-Us",
    repo_type="dataset",
    local_dir=raw_dir,
    # This tells it to skip any folder named "videos" and common video file types
    ignore_patterns=["**/videos/**", "**/*.mp4", "**/*.mkv", "**/*.webm"] 
)

print(f"Raw data saved to {raw_dir}")

