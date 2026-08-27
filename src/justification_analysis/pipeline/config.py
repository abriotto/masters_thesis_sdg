"""The single analysis configuration. Every input and output path derives here.

Before this module the string
`analysis/cross_model/base/voting/prompt_v4/justification_analysis` was
hard-coded independently in six places, and the model set was hard-coded in
two more. Pointing the pipeline at fine-tuned outputs meant editing all of
them and hoping none was missed.

Now there is one object:

    config = AnalysisConfig(stage="finetuned")

and every path follows from it.

THE RULE THAT MATTERS: switching stage must never silently fall back to
`base`. Every accessor that resolves an input either finds it under the
CONFIGURED stage or raises, naming the stage it looked for. There is no
default-to-base branch anywhere in this module, by construction.

Outputs live under their stage's own namespace, so a fine-tuned run cannot
overwrite a base artifact even by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_NAME = "masters_thesis_sdg"

BASE_STAGE = "base"
DEFAULT_PROMPT_VERSION = "prompt_v4"


def find_repo_root(start: Path = None, repo_name: str = REPO_NAME) -> Path:
    current = (start or Path.cwd()).resolve()
    while current.name != repo_name:
        if current.parent == current:
            raise FileNotFoundError(f"repo root {repo_name!r} not found")
        current = current.parent
    return current


@dataclass(frozen=True)
class ModelSpec:
    """One model in the analysis.

    `display` is what appears in tables and figures; `folder_glob` is how its
    per-model analysis directory is located under `analysis/`. Nothing
    downstream should hard-code either - statistics functions take the model
    order from the configuration.
    """
    display: str
    short: str
    folder_glob: str


DEFAULT_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec("Gemma 4 2B", "E2B", "*gemma-4-E2B*"),
    ModelSpec("Gemma 4 4B", "E4B", "*gemma-4-E4B*"),
    ModelSpec("Gemma 4 31B", "31B", "*gemma-4-31B*"),
)

# Structure of the decoding design. Stochastic runs are repeated realisations
# of the same games; greedy is one deterministic run. Kept here so a stage
# with a different run structure does not require code changes.
DEFAULT_STOCHASTIC_RUNS: Tuple[str, ...] = ("run_1", "run_2", "run_3")
DEFAULT_GREEDY_RUNS: Tuple[str, ...] = ("greedy_t0",)

VOTE_TABLE_RELATIVE = Path("vote_stability/tables/llm_vote_file_level.csv")


@dataclass(frozen=True)
class AnalysisConfig:
    """Which corpus is being analysed, and where everything for it lives."""

    stage: str = BASE_STAGE
    prompt_version: str = DEFAULT_PROMPT_VERSION
    models: Tuple[ModelSpec, ...] = DEFAULT_MODELS
    stochastic_runs: Tuple[str, ...] = DEFAULT_STOCHASTIC_RUNS
    greedy_runs: Tuple[str, ...] = DEFAULT_GREEDY_RUNS
    repo_root: Path = field(default_factory=find_repo_root)

    # -- vocabulary derived from the configured model set -------------------

    @property
    def model_order(self) -> List[str]:
        return [model.display for model in self.models]

    @property
    def model_labels(self) -> Dict[str, str]:
        return {model.display: model.short for model in self.models}

    @property
    def runs_by_decoding(self) -> Dict[str, Tuple[str, ...]]:
        return {"Stochastic": self.stochastic_runs, "Greedy": self.greedy_runs}

    @property
    def all_runs(self) -> Tuple[str, ...]:
        return tuple(self.stochastic_runs) + tuple(self.greedy_runs)

    @property
    def is_base(self) -> bool:
        return self.stage == BASE_STAGE

    # -- inputs -------------------------------------------------------------

    def model_analysis_dir(self, model: ModelSpec) -> Path:
        """The model's analysis folder, resolved by glob and required unique."""
        analysis_root = self.repo_root / "analysis"
        matches = sorted(
            folder for folder in analysis_root.glob(model.folder_glob)
            if folder.is_dir()
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one analysis folder matching "
                f"{model.folder_glob!r} for {model.display}, found {matches}"
            )
        return matches[0]

    def vote_table(self, model: ModelSpec) -> Path:
        """The per-model vote table FOR THE CONFIGURED STAGE.

        Raises rather than falling back. A missing fine-tuned table must stop
        the analysis, not quietly hand back base results.
        """
        path = (self.model_analysis_dir(model) / self.stage / "voting"
                / self.prompt_version / VOTE_TABLE_RELATIVE)
        if not path.exists():
            raise FileNotFoundError(
                f"no vote table for stage={self.stage!r}, "
                f"prompt={self.prompt_version!r}, model={model.display!r}:\n"
                f"  {path}\n"
                f"This stage has no corpus. The analysis stops here rather "
                f"than falling back to another stage."
            )
        return path

    def vote_tables(self) -> List[Tuple[ModelSpec, Path]]:
        return [(model, self.vote_table(model)) for model in self.models]

    def require_inputs(self) -> None:
        """Fail now, with every missing path listed, rather than mid-analysis."""
        missing = []
        for model in self.models:
            try:
                self.vote_table(model)
            except FileNotFoundError as error:
                missing.append(str(error))
        if missing:
            raise FileNotFoundError(
                f"stage {self.stage!r} is not analysable:\n\n"
                + "\n\n".join(missing)
            )

    # -- outputs ------------------------------------------------------------

    @property
    def artifact_root(self) -> Path:
        """Everything corpus-dependent for this stage lives under here."""
        return (self.repo_root / "analysis" / "cross_model" / self.stage
                / "voting" / self.prompt_version / "justification_analysis")

    @property
    def discourse_dir(self) -> Path:
        return self.artifact_root / "discourse_parser"

    @property
    def parser_candidates_path(self) -> Path:
        return self.discourse_dir / "discopy_explicit_candidates.csv"

    @property
    def parser_manifest_path(self) -> Path:
        return self.discourse_dir / "discopy_explicit_candidates.manifest.json"

    @property
    def dimlex_occurrences_path(self) -> Path:
        return self.artifact_root / "dimlex_analysis" / "dimlex_occurrences.csv"

    @property
    def dimlex_manifest_path(self) -> Path:
        return (self.artifact_root / "dimlex_analysis"
                / "dimlex_occurrences.manifest.json")

    @property
    def final_discourse_tables(self) -> Path:
        return self.discourse_dir / "thesis_tables" / "final_discourse"

    @property
    def final_discourse_figures(self) -> Path:
        return self.discourse_dir / "figures" / "final_discourse"

    @property
    def diagnostics_dir(self) -> Path:
        return self.discourse_dir / "diagnostics"

    @property
    def semantic_dir(self) -> Path:
        return self.artifact_root / "semantic_annotation"

    @property
    def joint_dir(self) -> Path:
        return self.artifact_root / "joint_discourse_semantic_justification"

    # -- provenance ---------------------------------------------------------

    def describe(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "prompt_version": self.prompt_version,
            "models": self.model_order,
            "stochastic_runs": list(self.stochastic_runs),
            "greedy_runs": list(self.greedy_runs),
            "artifact_root": str(self.artifact_root.relative_to(self.repo_root)),
        }

    def with_stage(self, stage: str) -> "AnalysisConfig":
        return replace(self, stage=stage)


def default_config(**overrides) -> AnalysisConfig:
    """The configuration a notebook gets unless it says otherwise.

    Defaults to the base stage because that is the frozen thesis corpus.
    Switching is a one-line change:

        config = default_config(stage="finetuned")
    """
    return AnalysisConfig(**overrides)
