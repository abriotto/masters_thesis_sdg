# Codebook: Epistemic Basis of LLM Vote Justifications in One Night Ultimate Werewolf

**Version 0.1 — for supervisor review**
Author: Anna Briotto · Date: 2026-08-06

---

## 1. What this measures

Each model produces a short public justification for its vote. This scheme codes **what the
model treated as a reason** — the epistemic basis it appealed to — in order to answer:

1. How much do models reason **deductively** from the rules and the game state?
2. How much do they defer to **other players' suspicions** (bandwagon / accusation-following)?
3. How much do they rely on **behavioural cues** (aggression, evasiveness, apparent dishonesty)?

**Scope claim.** This codes the justification as a communicative artifact, not the model's
internal computation. Justifications are not assumed to be faithful traces of reasoning
(cf. §2.4 of the progress report). The validity claim is that the *coded basis* covaries
with independently measured rule competence — a claim about text, and testable (§8).

---

## 2. Provenance

The scheme is derived, not authored. Each component has a source.

| component | source |
|---|---|
| Premise vs. inference gate | Toulmin: *data / warrant* vs. *claim*. A rule stated is a warrant; a rule used licenses a claim. |
| Epistemic / argumentative / social separation | Weinberger & Fischer (2006), CSCL framework for argumentative knowledge construction |
| `Social` bounded against speaker-side acts | Lai et al. (2023), *Werewolf Among Us* — 6 persuasion strategies (§7) |
| `Rule-Deduction`, `Consistency` definitions | adapted from the ONUW argument-graph annotation prompt (earlier thesis iteration) |
| Contents of `Other` | to be determined bottom-up from the pilot (§8) |
| IAA-driven guideline revision procedure | Visser, Lawrence, Reed, Wagemans & Walton, *Annotating Argument Schemes* |

---

## 3. Unit of analysis

**The sentence.**

Segmentation is by terminal punctuation. No annotator judgment is involved, so unitizing
agreement is 1.0 by construction.

*Cost of this choice:* sub-sentential claims are not separable, and a sentence may carry
more than one basis. This is handled by allowing **multiple labels per sentence** rather
than by finer segmentation.

*Why not claim-level:* free segmentation into claims requires arbitrary boundary decisions
("they give a consistent narrative" / "that contradicts Elliot") and cannot represent
unstated conclusions, which are common. Sentence units avoid both problems.

---

## 4. Two levels of coding

### 4.1 Justification level — three binary premise flags

Coded once per justification, not per sentence.

| flag | fires when the justification |
|---|---|
| `cites_rule` | states an ONUW rule, role ability, deck fact, or win condition |
| `cites_transcript` | reports what a player said or did in the game |
| `asserts_unsourced` | asserts a game-state fact that is neither a rule nor traceable to the transcript |

**Provenance, not correctness.** A fabricated swap presented as something that happened in
the game is still `cites_transcript`. Whether it actually happened is a separate variable,
so that reasoning type can be cross-tabulated against factual accuracy later.

**Derived measure — rule citation vs. rule use.** `cites_rule = 1` with zero `Deduction`
sentences is a justification that recites a rule and derives nothing from it. The gap
between citation rate and use rate is a primary result.

### 4.2 Sentence level — six reasoning labels, multi-label

A sentence receives a label only if it **draws a conclusion**. Sentences that only restate
the transcript, recite a rule, or restate the vote receive no label and are excluded from
the reasoning denominator.

---

## 5. The six labels

### `Deduction`
The conclusion follows because an ONUW rule, role ability, card movement, night ordering,
deck composition, or win condition **forces it or rules it out**.

> *"According to the rules the Robber takes on that role's alignment; therefore Elliot is
> now the Werewolf."*

**Test:** Given the stated premises, do the rules themselves force or exclude the
conclusion, without an additional guess about honesty, motive, or suspiciousness?

**Do not apply** merely because the sentence mentions roles, cards, swaps, or rules. A rule
stated without a conclusion drawn from it is `cites_rule` and nothing more.

---

### `Consistency`
The conclusion rests on **comparing statements** — they contradict, corroborate, align, or
one is left unsupported by the others.

> *"Ashley and Sukeshi provide a consistent narrative that contradicts Elliot's claims."*
> *"He claimed Doppelganger-Insomniac, but no one claimed Insomniac and he never said who
> he copied."*

**Test:** Can the conclusion be reached by comparing what was said, without applying a rule?

**Boundary — vs. `Behavioral`:** `Consistency` stops at *the accounts conflict*. If the
sentence takes the further step to *therefore he is lying / hiding something*, it is
**also** `Behavioral`. Both may apply.

---

### `Behavioral`
The conclusion rests on **how a player came across, or a judgment about their honesty,
motive, or strategy**: aggression, nervousness, hesitation, evasiveness, defensiveness,
bluffing, deflecting, manipulating, "typical Werewolf behaviour".

> *"His inconsistency and his pivot to suggesting he might be the Tanner are common
> behaviours for a Werewolf attempting to deflect suspicion."*

**Test:** Is the ground a claim about the *person* — their manner, credibility, or intent —
rather than about the game state?

---

### `Social`
**The reason is that other players suspect the target.** The model adopts someone else's
*judgment* as its own ground.

> *"Several players thought Hailey was acting strangely, so I vote Hailey."*
> *"Jack is convinced Mike is the Werewolf, and I agree."*
> *"The group has converged on Mike."*

**Test:** Strip out every other consideration. Is "someone else finds this player
suspicious" doing the work?

**Boundary — vs. `Testimony` (the critical one):**

| the model defers to | label |
|---|---|
| another player's **factual assertion** ("I'm the Seer", "I saw X's card", "I swapped with Y") | `Testimony` |
| another player's **judgment or suspicion** ("I think B is the wolf", "B is acting shady") | `Social` |

Fact vs. opinion. This is the distinction the whole research question turns on: accepting a
report is evidence-following; accepting a suspicion is being persuaded.

**Do not apply** when the justification *mentions* that others suspected someone and then
sets it aside. In *"others suspected Sukeshi and Sian, but the direct admissions are the
strongest evidence"* the suspicion is reported, not relied on — no `Social`.

---

### `Testimony`
One player's **factual claim or report** is accepted at face value as the ground.

> *"Ashley admitted to starting as the Werewolf."*
> *"The direct admissions from Ashley and Elliot are the strongest evidence."*

**Boundary:** if two or more accounts are compared for agreement, that is `Consistency`. If
the testimony is combined with a rule to derive a new game state, that is `Deduction`.

---

### `Other`
A conclusion is drawn, but on none of the above bases, or on no stated basis at all.

Includes: vote-by-elimination ("nobody else looks suspicious"), payoff reasoning ("voting
X maximises our chance of catching a wolf"), explicit uncertainty ("there isn't enough
evidence"), and bare assertion.

**This label is a diagnostic, not a category.** If `Other` exceeds ~15% of labelled
sentences in the pilot, its contents are inspected and the recurring pattern is promoted to
a named label (§8). Payoff reasoning is the most likely candidate — it appears in the first
justification sampled from `results/voting/`.

---

## 6. Decision procedure

For each sentence, in order:

1. **Does it draw a conclusion?** No → no label. (Restating the transcript, reciting a rule,
   restating the vote.) Yes → continue.
2. **Do the rules force it?** → `Deduction`
3. **Is the ground a comparison between statements?** → `Consistency`
4. **Is the ground someone else's suspicion or the group's opinion?** → `Social`
5. **Is the ground someone else's factual claim, taken at face value?** → `Testimony`
6. **Is the ground the player's manner, credibility, or motive?** → `Behavioral`
7. **None of these?** → `Other`

Steps 2–6 are not exclusive. Apply every label whose ground is present.

Code the reasoning **as presented**. Do not fact-check, and do not repair invalid reasoning.

---

## 7. Relation to Lai et al. (2023)

Lai's layer codes, per utterance, what a **speaker did**: Identity Declaration,
Interrogation, Evidence, Accusation, Defense, Call for Action.

This scheme codes what the **voting model treated as a reason**. It is the *listener-side
dual* of Lai's speaker-side scheme, over the same transcripts.

| this label | the kind of transcript utterance it leans on (Lai) |
|---|---|
| `Testimony` | Identity Declaration, Evidence |
| `Social` | Accusation |
| `Behavioral` | Defense, Interrogation — coded as *behaviour observed*, not as content |
| `Consistency` | a **relation between** several utterances; no single Lai label |
| `Deduction` | **no Lai counterpart** — rules are not utterances |
| — | Call for Action has no counterpart; it is a speaker move, not an evidence type |

Two things this buys:

- **Positioning.** The gap is explicit: Lai codes human speaker-side rhetoric; LLM
  conformity work measures behaviour without text; argumentation frameworks code basis but
  only in debate and classroom corpora. Nobody codes the epistemic basis of an LLM's own
  vote justification in a social deduction game.
- **Convergent validity, cheaply.** Do justifications coded `Social` sit on transcripts that
  are Accusation-dense in Lai's existing annotation layer? The data for this test is
  already in the repository.

**Note on constructs.** `Social` measures socially-grounded *justification*. It is not
conformity in the Asch sense, which is a behavioural claim about answer change under
majority pressure. Stating this prevents a reader conflating them. The causal version would
require an ablation — strip the accusatory turns from the transcript and measure vote shift
— and is out of scope here.

---

## 8. Validation plan

| step | n | output |
|---|---|---|
| 1. Pilot, single coder | 30 justifications | inspect `Other`; promote recurring categories; revise codebook |
| 2. Double-code, independent, blind to model and to vote correctness | 100 justifications, **stratified** | Krippendorff's α **per label**, with instance counts |
| 3. Freeze codebook | — | v1.0 |
| 4. LLM annotation of full corpus | ~2,300 justifications | label distribution per model |
| 5. LLM validated against held-out human gold | 100 justifications | LLM–human α per label, against human–human α as ceiling; check for systematic over-application |
| 6. Criterion validity | full corpus | see below |

**Report agreement on a fresh sample.** Step 2 must not reuse the justifications from step 1
that were used to revise the guidelines.

**Stratify the reliability sample.** `Social` is expected to be rare (plausibly 2–4% of
sentences). A random 100-sentence sample would contain ~3 instances and α for that label
would be noise — fatal, since `Social` is a headline construct. Oversample sentences flagged
by a keyword pre-filter as candidate `Social` / `Behavioral`, and report per-label α with
counts rather than one pooled figure.

**Criterion validity — the test that makes this validation rather than reliability.**
Pre-registered prediction:

> Deduction share rises with rule competence (questionnaire ordering 31B ≫ 4B ≈ 2B), while
> `Social` and `Behavioral` share fall.

If it holds, the scheme measures what it claims. If it does not, that is a substantive
finding in its own right, consistent with §4.5's result that some surface signals track
capability and others predict nothing.

**Fallback if no second coder is available:** intra-coder test–retest at ≥3 weeks. Weaker,
reportable, and to be named as a limitation rather than presented as equivalent.

---

## 9. Worked examples

### Example A

> "Ashley admitted to starting as the Werewolf, and Elliot admitted to being the Robber and
> swapping cards with Ashley. According to the game rules, the Robber swaps their card with
> another player and takes on that role's alignment and win conditions. Therefore, if their
> claims are true, Elliot is now the Werewolf. While other players like Sukeshi and Sian were
> suspected, the direct admissions from Ashley and Elliot provide the strongest evidence for
> Elliot being the current Werewolf."

Flags: `cites_rule` ✓ · `cites_transcript` ✓ · `asserts_unsourced` ✗

| sentence | label |
|---|---|
| Ashley admitted Werewolf; Elliot admitted Robber and swapping | `Testimony` |
| According to the rules, the Robber takes on that role's alignment | *no label — rule cited, nothing derived* |
| Therefore Elliot is now the Werewolf | `Deduction` |
| Others were suspected, but the direct admissions are the strongest evidence | `Testimony` |

Deduction share of labelled sentences = 1/3. Rule cited **and** used.

Note sentence 4: suspicion is mentioned and set aside, so **no `Social`**. Coding by content
rather than by ground would have scored this toward the bandwagon rate — the opposite of
what happened.

### Example B

> "Elliot claimed to be the Doppelganger who copied the Insomniac, but no other player
> claimed the Insomniac role, and he failed to specify which player he had copied. Ashley
> (claiming Seer) and Sukeshi (claiming Robber) provide a consistent narrative that
> contradicts Elliot's claims. Furthermore, Elliot's inconsistency and his eventual pivot to
> suggesting he might be the Tanner are common behaviors for a Werewolf attempting to deflect
> suspicion."

Flags: `cites_rule` ✗ · `cites_transcript` ✓ · `asserts_unsourced` ✗

| sentence | label |
|---|---|
| Claimed Doppelganger-Insomniac, but no one claimed Insomniac and he didn't say who he copied | `Consistency` |
| Ashley and Sukeshi give a consistent narrative that contradicts Elliot | `Consistency` |
| His inconsistency and pivot to Tanner are common Werewolf deflection behaviours | `Behavioral` |

Deduction share = 0. No rule cited despite a Doppelganger claim being on the table —
precisely the citation/use gap the flags are designed to surface.

---

## 10. Decisions requested from supervisor

1. **Sentence as unit** — accept the loss of sub-sentential claims in exchange for perfect
   unitizing agreement and a tractable annotation load (~8,000 units)?
2. **Multi-label** — allow a sentence to carry both `Consistency` and `Behavioral`, or force
   a single primary label for cleaner proportions?
3. **`Other` as a holding category** — promote its contents after the pilot, or pre-commit
   now to `Payoff-Reasoning`, `Elimination-Default`, and `Uncertainty` as named labels?
4. **Second coder** — is one available for step 2, or is test–retest the realistic fallback?
5. **Scope of the criterion test** — is the questionnaire proficiency ordering an acceptable
   external criterion, given it was designed as a gate rather than a graded scale (§2.1)?

---

## References

- Lai, B. et al. (2023). *Werewolf Among Us: Multimodal Resources for Modeling Persuasion
  Behaviors in Social Deduction Games.* Findings of ACL 2023.
- Weinberger, A. & Fischer, F. (2006). *A framework to analyze argumentative knowledge
  construction in computer-supported collaborative learning.* Computers & Education, 46(1).
- Visser, J., Lawrence, J., Reed, C., Wagemans, J. & Walton, D. *Annotating Argument
  Schemes.* Argumentation.
- Toulmin, S. (1958). *The Uses of Argument.* Cambridge University Press.
- Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology.* 2nd ed.
