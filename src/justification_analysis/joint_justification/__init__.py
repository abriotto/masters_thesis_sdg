"""Joint discourse x semantic analysis at the JUSTIFICATION level.

Supersedes the sentence-level joint analysis in
`src.justification_analysis.joint`, which is deprecated for thesis purposes:
same-sentence co-occurrence misses relations that connect content across a
sentence boundary, and penalises a model for splitting one thought into two
sentences.
"""
