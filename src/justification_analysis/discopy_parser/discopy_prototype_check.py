"""Minimal discopy prototype on manually constructed ONUW-style sentences.

Prints raw parser output: candidate spans, accept/reject, sense labels,
character offsets, discontinuity, and confidence.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg"
sys.path.insert(0, REPO)

MODEL_PATH = sys.argv[1]
BERT_MODEL = sys.argv[2] if len(sys.argv) > 2 else "bert-base-cased"

SENTENCES = [
    "Alice is suspicious because her claim is impossible.",
    "Alice claimed Seer, but Bob contradicted her.",
    "Alice acted as the Robber.",
    "As Bob acted first, Alice's story cannot be true.",
    "For Alice, Bob is the obvious suspect.",
    "Bob lied, for his account contradicts Alice's.",
    "Alice claimed Seer and Bob claimed Robber.",
    "If Alice is the Robber, then Bob cannot still hold that card.",
    # extra discontinuous / ONUW-domain probes
    "Either Alice or Bob must be the Werewolf.",
    "Alice swapped with Bob and then looked at the center card.",
]


def main():
    import tensorflow as tf
    import transformers
    from src.utils.sentences import split_sentences
    from src.justification_analysis.discopy_parser.discopy_explicit import (
        build_document, load_explicit_component, parse_explicit,
        verify_against_upstream, collapse_pdtb_sense,
    )
    from discopy_data.nn.bert import get_sentence_embedder

    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)
    print("python      ", sys.version.split()[0])
    print("tensorflow  ", tf.__version__)
    print("transformers", transformers.__version__)
    print("checkpoint  ", os.path.basename(MODEL_PATH))
    print("backbone    ", BERT_MODEL)

    embedder = get_sentence_embedder(BERT_MODEL)
    component = load_explicit_component(MODEL_PATH)
    print("component   ", type(component).__name__)
    print("classes     ", component.classes)
    print("used_context", component.used_context)

    text = " ".join(SENTENCES)
    sentences = split_sentences(text)
    print(f"\ndeterministic segmentation -> {len(sentences)} sentences "
          f"(input had {len(SENTENCES)})")

    doc = build_document("prototype", text, sentences)
    print(f"document: {len(doc.sentences)} sentences, {len(doc.get_tokens())} tokens")
    for s_i, s in enumerate(doc.sentences):
        toks = s.tokens
        print(f"  sent[{s_i}] offs=({toks[0].offset_begin},{toks[-1].offset_end}) "
              f"{text[toks[0].offset_begin:toks[-1].offset_end][:70]!r}")

    for s_i, s in enumerate(doc.sentences):
        doc.sentences[s_i].embeddings = embedder(s.tokens)

    n_up, n_batch = verify_against_upstream(component, doc)
    print(f"\nbatched parse == upstream parse: {n_up} relations both ways  [OK]")

    rows = parse_explicit(component, doc, keep_nosense=True)

    print("\n" + "=" * 80)
    print(f"ALL CONNECTIVE CANDIDATES ({len(rows)}), including rejected ones")
    print("=" * 80)
    print(f"{'cand':<16} {'sent':>4} {'accepted':>9} {'raw sense':<24} "
          f"{'top level':<12} {'conf':>6}  spans")
    for r in sorted(rows, key=lambda r: (r["sentence_index"], r["char_spans"])):
        print(f"{r['candidate_surface']:<16} {r['sentence_index']:>4} "
              f"{str(r['is_connective']):>9} {r['raw_sense']:<24} "
              f"{str(r['top_level']):<12} {r['confidence']:>6.3f}  "
              f"{r['char_spans']}"
              f"{'  <-- DISCONTINUOUS' if r['is_discontinuous'] else ''}")

    print("\n" + "=" * 80)
    print("PER-SENTENCE READING")
    print("=" * 80)
    for s_i, sent_text in enumerate(sentences):
        print(f"\n[{s_i}] {sent_text}")
        for r in rows:
            if r["sentence_index"] != s_i:
                continue
            verdict = ("CONNECTIVE -> " + str(r["top_level"])
                       if r["is_connective"] else "NOT a connective")
            surface = " ... ".join(
                text[a:b] for a, b in r["char_spans"]
            )
            print(f"      {surface!r:<24} {verdict:<28} "
                  f"({r['raw_sense']}, p={r['confidence']:.3f})")

    print("\n" + "=" * 80)
    print("SENSE LABEL INVENTORY -> TOP LEVEL")
    print("=" * 80)
    for cls in component.classes:
        print(f"  {cls:<28} -> {collapse_pdtb_sense(cls)}")


if __name__ == "__main__":
    main()
