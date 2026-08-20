"""
Sentence segmentation for LLM vote justifications.

The single definition of a sentence for this project. Imported by:

  * src/utils/experiment_utils.py     generation-time soft warnings
                                      (run_llm_votes.py, run_eval.py)
  * notebooks/04_justification_analysis/1_dimlex_justification_analysis.ipynb
                                      DiMLex marker rates, justification length
  * src/pt_annotation/               the DeepSeek justification annotator

Everything downstream must agree on what a sentence is. The annotation prompt
(src/prompts/justification_annotation.txt) takes sentences as fixed input and
keys every annotation to a sentence_id, so if the reported justification
length and the annotated units came from different splitters, sentence-level
results from the two pipelines would not line up.

The rule is terminal punctuation, not a trained splitter. Measured on all
2,287 prompt_v4 justifications, an abbreviation-aware splitter (nltk punkt,
spacy senter) would change 42 of them and nothing else -- the corpus has no
decimals, no initials and no newlines. Handling abbreviations explicitly, as
below, gets that result with a rule that can be written down and audited,
without adding a dependency or a model download.
"""

import re

SENTENCE_PATTERN = re.compile(
    r"[^.!?]+(?:[.!?]+|$)",
    flags=re.UNICODE,
)

WORD_PATTERN = re.compile(r"\w", flags=re.UNICODE)

# Non-terminal periods. Measured against all 2,287 prompt_v4 justifications,
# these are the ONLY source of divergence between this rule and an
# abbreviation-aware splitter: 42 justifications (1.84%), 77 spurious units
# out of 8,105 (0.95%). The corpus contains no decimal numbers, no initials
# and no newlines, which is why a trained splitter (punkt, spacy senter) buys
# nothing further here -- and why it is not worth moving unitizing decisions
# into an opaque model that the write-up cannot describe.
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|cf\.|Mr\.|Mrs\.|Ms\.|Dr\.|St\.|approx\.)",
    flags=re.IGNORECASE,
)

# A closing quote or bracket after terminal punctuation belongs to the
# sentence it closes, not to the next one. Without this, '...the werewolf."
# While the swaps...' hands the annotator a unit that opens with a stray
# quotation mark, and every evidence_span drawn from it inherits the mark.
TRAILING_CLOSER_PATTERN = re.compile(r'^["\'”’\)\]]+')

# Private-use code point: it cannot occur in the source text, so masking
# and unmasking are exactly reversible and every returned sentence is
# verbatim -- which the prompt requires, since evidence_span must be an
# exact substring of the sentence it is drawn from.
_PROTECTED_PERIOD = ""


def split_sentences(text):
    """Split a justification into sentences on terminal punctuation.

    Chunks with no word character (stray punctuation, whitespace) are
    dropped: they are not units, and passing them to the annotator would
    invite empty annotations that look like real coding decisions.

    Semicolons and colons do not split. The annotation prompt's worked
    example keeps a semicolon-joined sentence as one unit and gives it two
    annotations, so more than one basis inside a sentence is represented by
    multi-labelling rather than by finer segmentation.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    masked = ABBREVIATION_PATTERN.sub(
        lambda match: match.group(0).replace(".", _PROTECTED_PERIOD),
        text.strip(),
    )

    chunks = [
        chunk.replace(_PROTECTED_PERIOD, ".")
        for chunk in SENTENCE_PATTERN.findall(masked)
    ]

    sentences = []
    for chunk in chunks:
        chunk = chunk.strip()

        if WORD_PATTERN.search(chunk):
            closer = TRAILING_CLOSER_PATTERN.match(chunk)
            if closer and sentences:
                sentences[-1] = sentences[-1] + closer.group(0)
                chunk = chunk[closer.end():].strip()
            sentences.append(chunk)
        elif sentences:
            # A chunk with no word character is not a unit, but it is still
            # text: "...so...')." splits into "...so..." and "')." and the
            # tail must rejoin the sentence it terminates rather than be
            # dropped. Concatenating the units must reproduce the
            # justification exactly, or the annotated corpus and the source
            # corpus quietly disagree.
            sentences[-1] = sentences[-1] + chunk

    return sentences


def count_sentences(text):
    return len(split_sentences(text))


def build_sentence_records(text):
    """Sentences as the annotation prompt expects them: 1-indexed
    `sentence_id` plus verbatim `text`."""
    return [
        {"sentence_id": index, "text": sentence}
        for index, sentence in enumerate(split_sentences(text), start=1)
    ]
