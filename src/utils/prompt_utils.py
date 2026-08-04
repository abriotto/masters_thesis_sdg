from __future__ import annotations

"""
Shared prompt assembly.

The finetuning prompts and the voting prompts must be assembled identically, or the
familiarisation transfer is confounded by a formatting difference rather than by what
the model learned. This module is the single source of truth for that layout, imported
by both src/voting/run_llm_votes.py and src/finetuning/build_sft_dataset.py.

Changing the layout here changes prompts on both sides. Any already-collected results
were produced with the layout as of the run that generated them.
"""


def build_full_prompt(
    base_prompt: str,
    rules_text: str,
    player_names: list[str],
    transcript_text: str,
) -> str:
    players_str = ", ".join(player_names)

    return f"""{base_prompt}

Here are the game rules:

{rules_text}

## Player list

{players_str}

## Transcript

{transcript_text}
""".strip()


__all__ = ["build_full_prompt"]
