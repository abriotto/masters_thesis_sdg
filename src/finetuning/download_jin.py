from __future__ import annotations

"""
Download the Jin et al. ONUW GPT-4 dataset.

Source: https://github.com/KylJin/Werewolf (dataset_process/gpt4_dataset)

The upstream directory holds `episode_001.json` ... `episode_120.json`. Each file
is a full One Night Ultimate Werewolf game played by GPT-4 agents:

    {
      "messages": [{agent_name, content, turn, visible_to, belief, thought, strategy}, ...],
      "evaluation": {roles_assigned, roles_ground_truth, role_pool, player_backends,
                     voting_result, winner}
    }

Note on the `evaluation` keys, which are easy to read backwards:
- `roles_assigned`      = STARTING roles (what each player was dealt).
- `roles_ground_truth`  = FINAL roles (after the night swaps), i.e. what decides the winner.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import requests

from src.utils.io_utils import find_repo_root


RAW_BASE_URL = (
    "https://raw.githubusercontent.com/KylJin/Werewolf/main/dataset_process/gpt4_dataset"
)

# Known upstream size as of 2026-08-04. The script probes one past this to warn
# if the dataset has grown since.
EXPECTED_NUM_EPISODES = 120

REQUIRED_EVALUATION_KEYS = {
    "roles_assigned",
    "roles_ground_truth",
    "role_pool",
    "voting_result",
    "winner",
}


def enable_system_trust_store() -> bool:
    """
    Use the OS certificate store instead of certifi's bundle.

    Needed on Windows machines behind a TLS-inspecting proxy, where certifi cannot
    build a chain to raw.githubusercontent.com. No-op elsewhere, and unlike
    `verify=False` it still verifies the certificate.
    """
    try:
        import ssl

        import truststore

        truststore.inject_into_ssl()
        _ = ssl  # imported for clarity about what is being patched
        return True
    except Exception:
        return False


def episode_name(index: int) -> str:
    return f"episode_{index:03d}.json"


def validate_episode(payload: Any) -> list[str]:
    """Return a list of structural problems; empty means the episode looks sane."""
    problems: list[str] = []

    if not isinstance(payload, dict):
        return ["top_level_not_object"]

    messages = payload.get("messages")
    evaluation = payload.get("evaluation")

    if not isinstance(messages, list) or not messages:
        problems.append("missing_or_empty_messages")
    else:
        for field in ("agent_name", "content", "turn", "visible_to"):
            if any(field not in m for m in messages):
                problems.append(f"messages_missing_field:{field}")

    if not isinstance(evaluation, dict):
        problems.append("missing_evaluation")
    else:
        missing = sorted(REQUIRED_EVALUATION_KEYS - set(evaluation))
        if missing:
            problems.append(f"evaluation_missing_keys:{missing}")

    return problems


def fetch_episode(
    session: requests.Session,
    index: int,
    base_url: str,
    timeout: float,
    retries: int,
) -> Optional[Any]:
    """Return the parsed episode, or None if upstream returned 404."""
    url = f"{base_url}/{episode_name(index)}"
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 10.0))

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Jin et al. ONUW GPT-4 episodes into data/raw/."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/jin2024_onuw/gpt4_dataset",
        help="Destination directory, relative to the repo root.",
    )
    parser.add_argument("--base_url", type=str, default=RAW_BASE_URL)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=EXPECTED_NUM_EPISODES,
        help="How many episodes to request (episode_001 .. episode_N).",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--ca_bundle",
        type=str,
        default=None,
        help="Path to a CA bundle, if certifi and the system store both fail.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download episodes that are already present and valid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    used_system_trust = enable_system_trust_store()
    session = requests.Session()
    if args.ca_bundle:
        session.verify = args.ca_bundle
    print(
        f"TLS trust: {'system store (truststore)' if used_system_trust else 'certifi'}"
        f"{' + --ca_bundle override' if args.ca_bundle else ''}",
        flush=True,
    )

    downloaded: list[str] = []
    skipped: list[str] = []
    missing: list[int] = []
    invalid: dict[str, list[str]] = {}

    for index in range(1, args.num_episodes + 1):
        name = episode_name(index)
        output_path = output_dir / name

        if output_path.exists() and not args.overwrite:
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                problems = validate_episode(existing)
                if not problems:
                    skipped.append(name)
                    continue
                print(f"[{index}] re-downloading {name}: on-disk copy invalid ({problems})")
            except json.JSONDecodeError:
                print(f"[{index}] re-downloading {name}: on-disk copy is not valid JSON")

        payload = fetch_episode(
            session=session,
            index=index,
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
        )

        if payload is None:
            missing.append(index)
            print(f"[{index}] MISSING - {name} returned 404")
            continue

        problems = validate_episode(payload)
        if problems:
            invalid[name] = problems
            print(f"[{index}] INVALID - {name}: {problems}")

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        downloaded.append(name)
        print(f"[{index}/{args.num_episodes}] OK - {name}")

    # Probe one past the requested range so we notice if upstream grew.
    probe_index = args.num_episodes + 1
    probe = fetch_episode(
        session=session,
        index=probe_index,
        base_url=args.base_url,
        timeout=args.timeout,
        retries=1,
    )
    if probe is not None:
        print(
            f"\nWARNING: {episode_name(probe_index)} also exists upstream. "
            f"The dataset is larger than --num_episodes={args.num_episodes}; "
            "re-run with a higher value."
        )

    manifest = {
        "base_url": args.base_url,
        "requested_num_episodes": args.num_episodes,
        "num_downloaded": len(downloaded),
        "num_skipped_already_present": len(skipped),
        "missing_indices": missing,
        "invalid_episodes": invalid,
        "upstream_has_more_than_requested": probe is not None,
        "output_dir": args.output_dir,
    }
    (output_dir.parent / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    on_disk = sorted(p.name for p in output_dir.glob("episode_*.json"))
    print(
        f"\nDownloaded {len(downloaded)}, skipped {len(skipped)} already present, "
        f"{len(missing)} missing, {len(invalid)} structurally invalid."
    )
    print(f"Episodes on disk: {len(on_disk)} -> {output_dir}")


if __name__ == "__main__":
    main()
