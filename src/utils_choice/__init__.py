"""Discrete-choice models of who a voter targets in One Night Ultimate Werewolf.

Shared by the LLM surrogate notebook and the human-vote notebook, so both use
the same estimator, features, cross-validation and inference, and differ only
in whose votes they explain.
"""

from .features import (ALL_FEATURES, BLOCK_A, BLOCK_B, BLOCKS, CIRCLE_FEATURE,
                       CIRCLE_OPTION, PLAYER_FEATURES, build_ballot, ckey,
                       cols_for, load_accusation_features,
                       load_identity_claim_features, load_technique_features,
                       temporal)
from .model import ConditionalLogit, run_validation_checks
from .evaluation import (CV_SEEDS, LEARNERS, N_FOLDS, build_frame,
                         choice_metrics, cross_validate, reference_points,
                         run_grid)
from .inference import (block_bootstrap, block_test, bootstrap_paired_diff,
                        coef_table, fit_with_vcov, iia_test,
                        oof_loglik_by_game, stability_selection,
                        STABILITY_THRESHOLD)
from .io import (crowd_modal_map, human_vote_targets, llm_vote_targets,
                 load_vote_tables, save_tables, shares_from_targets,
                 GREEDY_RUN, STOCHASTIC_RUNS)

__all__ = [n for n in dir() if not n.startswith("_")]
