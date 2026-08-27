"""Artifact manifests, and the freshness gate that refuses stale ones.

The failure this exists to make impossible: an expensive parser artifact is
cached, the corpus underneath it changes (or the analysis is pointed at a
different stage), and the analysis happily consumes the old CSV because the
path still resolves. Nothing in the numbers looks wrong. The results are
silently about a corpus that no longer exists.

The gate is deliberately blunt:

  * an artifact is usable only if its manifest's corpus fingerprint EQUALS the
    fingerprint of the corpus currently loaded;
  * a mismatch raises `StaleArtifactError`. It is not a warning, there is no
    `force` parameter, and there is no fallback to another stage's artifact;
  * a missing artifact or a missing manifest raises the same way;
  * every failure prints the exact command that regenerates the artifact for
    the ACTIVE configuration.

Path equality is never treated as evidence of freshness. Two corpora can
occupy the same path at different times; only the fingerprint speaks.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.justification_analysis.pipeline.config import AnalysisConfig
from src.justification_analysis.pipeline import corpus as corpus_module

MANIFEST_VERSION = "1"


class StaleArtifactError(RuntimeError):
    """A cached artifact does not belong to the corpus being analysed."""


class MissingArtifactError(FileNotFoundError):
    """A required artifact, or its manifest, does not exist for this stage."""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def build_manifest(config: AnalysisConfig, corpus: pd.DataFrame,
                   artifact_path: Path, artifact_kind: str,
                   producer: Dict[str, object],
                   outputs: Dict[str, object] = None) -> Dict:
    """Everything needed to decide later whether this artifact is still valid."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "artifact_kind": artifact_kind,
        "artifact_file": Path(artifact_path).name,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus_module.corpus_summary(corpus, config),
        "config": config.describe(),
        "producer": producer,
        "outputs": outputs or {},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def write_manifest(manifest: Dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def read_manifest(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise MissingArtifactError(f"no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def regeneration_command(config: AnalysisConfig, artifact_kind: str) -> str:
    """The exact command that rebuilds this artifact for the ACTIVE config."""
    if artifact_kind == "discopy_explicit_candidates":
        return (
            "# discopy inference runs in its own environment (TensorFlow +\n"
            "# numpy<2), not the project's sdglogs env:\n"
            '"$DISCOPY_PYTHON" '
            "src/justification_analysis/discopy_parser/"
            "run_discopy_on_justifications.py \\\n"
            '    --model-path "$DISCOPY_CHECKPOINT" \\\n'
            f"    --stage {config.stage} \\\n"
            f"    --prompt-version {config.prompt_version}"
        )
    if artifact_kind == "dimlex_occurrences":
        return (
            "python src/justification_analysis/dimlex/build_dimlex_occurrences.py"
            f" --stage {config.stage} --prompt-version {config.prompt_version}"
        )
    return f"# no registered regeneration command for {artifact_kind!r}"


def _explain(config: AnalysisConfig, artifact_kind: str, artifact_path: Path,
             reason: str, detail: str = "") -> str:
    lines = [
        reason,
        "",
        f"  stage            : {config.stage}",
        f"  prompt version   : {config.prompt_version}",
        f"  artifact         : {artifact_path}",
    ]
    if detail:
        lines += ["", detail]
    lines += [
        "",
        "The analysis stops here. It will not fall back to another stage's",
        "artifact, and it does not treat the path resolving as evidence that",
        "the artifact belongs to this corpus.",
        "",
        "Regenerate with:",
        "",
        regeneration_command(config, artifact_kind),
    ]
    return "\n".join(lines)


def verify_artifact(config: AnalysisConfig, corpus: pd.DataFrame,
                    artifact_path: Path, manifest_path: Path,
                    artifact_kind: str) -> Dict:
    """Return the manifest, or refuse. There is no third outcome.

    No `force`, no `allow_stale`, no warning-and-continue: the whole point is
    that a caller cannot opt out of the check by accident.
    """
    artifact_path = Path(artifact_path)
    manifest_path = Path(manifest_path)

    if not artifact_path.exists():
        raise MissingArtifactError(_explain(
            config, artifact_kind, artifact_path,
            f"No {artifact_kind} artifact exists for stage "
            f"{config.stage!r}."))

    if not manifest_path.exists():
        raise MissingArtifactError(_explain(
            config, artifact_kind, artifact_path,
            f"The {artifact_kind} artifact exists but has no manifest, so "
            f"there is no way to tell which corpus it was built from.",
            f"  expected manifest: {manifest_path}"))

    manifest = read_manifest(manifest_path)
    recorded = manifest.get("corpus", {}).get("fingerprint")
    current = corpus_module.corpus_fingerprint(corpus)

    if recorded != current:
        recorded_corpus = manifest.get("corpus", {})
        detail = "\n".join([
            f"  artifact corpus  : {recorded}",
            f"                     stage={recorded_corpus.get('stage')!r} "
            f"n={recorded_corpus.get('n_justifications')} "
            f"games={recorded_corpus.get('n_games')}",
            f"  current corpus   : {current}",
            f"                     stage={config.stage!r} n={len(corpus)} "
            f"games={corpus['game_id'].nunique()}",
        ])
        raise StaleArtifactError(_explain(
            config, artifact_kind, artifact_path,
            f"STALE ARTIFACT: the cached {artifact_kind} was built from a "
            f"different corpus than the one now loaded.", detail))

    recorded_stage = manifest.get("config", {}).get("stage")
    if recorded_stage != config.stage:
        raise StaleArtifactError(_explain(
            config, artifact_kind, artifact_path,
            f"Manifest stage {recorded_stage!r} does not match the active "
            f"stage {config.stage!r}, even though the fingerprints agree.",
            "This means two stages share an identical corpus, which is almost "
            "certainly a configuration mistake."))

    return manifest


def load_verified_candidates(config: AnalysisConfig,
                             corpus: pd.DataFrame) -> tuple:
    """The parser candidate table, only if it belongs to this corpus."""
    manifest = verify_artifact(
        config, corpus,
        config.parser_candidates_path, config.parser_manifest_path,
        "discopy_explicit_candidates")
    candidates = pd.read_csv(config.parser_candidates_path)

    recorded = manifest.get("outputs", {}).get("n_candidates")
    if recorded is not None and int(recorded) != len(candidates):
        raise StaleArtifactError(
            f"the candidate file holds {len(candidates)} rows but its "
            f"manifest records {recorded}; the file changed after the "
            f"manifest was written.")
    return candidates, manifest


def provenance_block(config: AnalysisConfig, corpus: pd.DataFrame,
                     manifest: Optional[Dict] = None) -> str:
    """The header every notebook prints before doing anything."""
    summary = corpus_module.corpus_summary(corpus, config)
    lines = [
        "=" * 72,
        f"  STAGE            : {config.stage}",
        f"  PROMPT VERSION   : {config.prompt_version}",
        f"  MODELS           : {', '.join(config.model_order)}",
        f"  RUNS             : stochastic {', '.join(config.stochastic_runs)}"
        f" | greedy {', '.join(config.greedy_runs)}",
        f"  JUSTIFICATIONS   : {summary['n_justifications']:,}",
        f"  GAMES            : {summary['n_games']:,}",
        f"  SENTENCES        : {summary['n_sentences']:,}",
        f"  WORDS            : {summary['n_words']:,}",
        f"  CORPUS HASH      : {summary['fingerprint']}",
        f"  ARTIFACT DIR     : "
        f"{config.artifact_root.relative_to(config.repo_root)}",
    ]
    if manifest:
        producer = manifest.get("producer", {})
        lines += [
            "-" * 72,
            f"  ARTIFACT         : {manifest.get('artifact_file')}",
            f"  BUILT            : {manifest.get('written_at_utc')}",
            f"  PARSER           : {producer.get('implementation', '?')} "
            f"{producer.get('version', '')}".rstrip(),
            f"  CHECKPOINT       : {producer.get('checkpoint', '?')}",
            f"  BACKBONE         : {producer.get('bert_model', '?')}",
            f"  FRESHNESS        : verified against the current corpus hash",
        ]
    lines.append("=" * 72)
    return "\n".join(lines)
