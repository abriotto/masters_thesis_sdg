"""
Loaders for the raw Lai et al. 2023 ONUW release.

Deliberately minimal: this package now only reads the released files --
the annotation split JSONs and the per-session vote-outcome records -- and
canonicalises the session/game identifiers used to join them.

Everything that models human votes lives in :mod:`utils_choice`:

* ``utils_choice.human_lai`` -- the Lai et al. replication
* ``utils_choice.human_pairs`` -- the thesis pairwise dataset
* ``utils_choice.human_model`` -- grouped nested CV and the bootstrap

The earlier feature-extraction, row-building and evaluation helpers that
used to live here were superseded by those modules (and by the shared
``utils_choice.rq1_*`` feature pipeline) and have been removed.
"""
