"""Shared pipeline plumbing for the justification analyses.

Three concerns, one implementation each:

  * `config`   - the ONE analysis configuration (stage, prompt version, model
                 set, decoding structure) from which every input and output
                 path derives;
  * `corpus`   - the ONE canonical corpus loader, plus the deterministic
                 fingerprint that identifies which corpus an artifact belongs
                 to;
  * `manifest` - writing, reading and VERIFYING artifact manifests, so a
                 cached parser output can never be silently reused against a
                 corpus it was not produced from.
"""
