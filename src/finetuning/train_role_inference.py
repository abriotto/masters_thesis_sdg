from __future__ import annotations

"""
QLoRA finetuning for the ONUW role-inference familiarisation task.

Follows the Unsloth reference recipe for this checkpoint
(unslothai/notebooks -> nb/Gemma4_(31B)-Text.ipynb): FastModel + get_peft_model with
finetune_vision_layers=False, the "gemma-4-thinking" chat template, and
train_on_responses_only masking at the <|turn>model marker.

Two deviations from the reference, both deliberate:
- num_train_epochs instead of max_steps, because the dataset is small and fixed.
- save_strategy="epoch", so one training run yields several adapters to evaluate.
  There is no budget for a second run, so intermediate checkpoints are the only
  fallback if the last epoch turns out to have over-trained.

TWO DATASET MODES
-----------------
1. Answer-only (build_sft_dataset.py). Targets are `{"roles": {...}}` with no thought
   block, loss anchored at `<|turn>model\\n`. MEASURED TO SUPPRESS REASONING: after 25
   steps the model dropped from ~6,360 output tokens with a full thought block to ~130
   with none on the voting prompt, with thinking explicitly enabled. Every sequence
   went `<|turn>model` -> `{`, so it learned to skip the thought channel. Kept for
   reference and as the documented control, not recommended.

2. Traced (build_traced_dataset.py). The assistant turn is
   `<|channel>thought {base-model reasoning} <channel|> {gold answer}`, with
   --response_part '<channel|>' so the thought is unsupervised context and only the
   gold answer is trained, and --enable_thinking_in_prompt so the prompt matches the
   voting evaluation. Thinking never leaves the training distribution, and no gradient
   shapes the reasoning that is later measured. This also satisfies Unsloth's guidance
   to keep >=75% reasoning-style examples when finetuning Gemma 4.

Run --dry_run (tokenizer only, no GPU) and then --inspect_only (real collator) and read
what is actually inside the loss span before committing the run. For mode 2 the
supervised span must be the gold JSON alone, with the thought block masked.
"""

import unsloth  # noqa: F401  - must precede transformers/trl imports

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from src.utils.io_utils import find_repo_root
from src.utils.model_utils import load_local_model_for_training


INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA finetuning for ONUW role inference (Jin et al. transcripts)."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        help="Must match the base checkpoint used for the voting evaluation.",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/jin2024_onuw/sft_role_inference/train.jsonl",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/jin2024_onuw/sft_role_inference/val.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/finetuned/gemma-4-31B-role-inference-v1",
    )
    parser.add_argument("--chat_template", type=str, default="gemma-4-thinking")
    parser.add_argument(
        "--response_part",
        type=str,
        default=RESPONSE_PART,
        help=(
            "Where the loss starts. Default '<|turn>model\\n' supervises the whole "
            "assistant turn. For the traced dataset pass '<channel|>' so the thought "
            "block stays unsupervised context and only the gold answer is trained."
        ),
    )
    parser.add_argument(
        "--enable_thinking_in_prompt",
        action="store_true",
        help=(
            "Render training prompts in thinking-ON format, matching the voting "
            "evaluation. Only meaningful when the targets contain a thought block - "
            "pairing it with answer-only targets trains the model to ignore the "
            "thinking instruction."
        ),
    )

    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Overrides num_train_epochs when > 0. Use a small value to smoke-test the loop.",
    )
    parser.add_argument(
        "--limit_train_examples",
        type=int,
        default=-1,
        help="Truncate the training set. For smoke tests only.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument(
        "--eval_strategy",
        type=str,
        choices=["no", "epoch", "steps"],
        default="no",
        help=(
            "OFF by default. HF's eval path casts full logits to fp32 - "
            "[batch, seq_len, vocab] with a ~262k vocab - which OOMs at long "
            "sequence lengths even though training itself fits comfortably "
            "(Unsloth uses fused cross-entropy and never materialises them). "
            "Val loss is diluted cross-entropy over ~46 tokens and should not "
            "drive checkpoint selection anyway; use role accuracy and the voting "
            "probe instead."
        ),
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Tokenizer only, no GPU: render one example and show the approximate loss span.",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Full pipeline up to the real collator, print the exact loss span, then exit.",
    )
    parser.add_argument(
        "--smoke_test_samples",
        type=int,
        default=3,
        help="After training, generate on N val examples to confirm the thought channel survived.",
    )
    parser.add_argument(
        "--smoke_test_max_new_tokens",
        type=int,
        default=10000,
        help=(
            "Must be large enough for a full thought block plus the answer. The voting "
            "runs use 10000 and thought blocks alone reach ~7700 tokens; a small budget "
            "truncates mid-thought, which breaks the parser and looks like suppression."
        ),
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


THOUGHT_MARKER = "channel>thought"
TURN_END = "<turn|>\n"


def render_traced_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool,
) -> str:
    """
    Render a training example whose assistant turn contains a thought block.

    apply_chat_template CANNOT be used for this. The gemma-4-thinking template strips
    thought blocks out of completed assistant turns - Unsloth's guidance is that
    thoughts are not carried in conversation history - so passing
    "<|channel>thought ... <channel|>{answer}" as message content silently yields just
    the answer, byte-identical to an answer-only example.

    Instead: render the PROMPT with add_generation_prompt=True (exactly the string the
    model sees at inference, ending at "<|turn>model\\n"), then append the assistant
    text verbatim and close the turn. The sequence is then

        <|turn>user ...<turn|>\\n<|turn>model\\n<|channel>thought ...<channel|>{answer}<turn|>\\n

    which is what train_on_responses_only(response_part="<channel|>") expects.
    """
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if assistant is None:
        raise ValueError("Traced example has no assistant message.")

    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, **kwargs, enable_thinking=enable_thinking
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(prompt_messages, **kwargs)

    return (prompt_text + assistant["content"] + TURN_END).removeprefix("<bos>")


def has_thought_block(content: str, response_part: str) -> bool:
    """
    Whether an assistant turn carries a thought block, for any checkpoint.

    A non-default response_part IS the closing thought delimiter, so its presence in
    the content identifies a traced target without knowing the model's token names.
    """
    if response_part and response_part != RESPONSE_PART and response_part in content:
        return True
    return THOUGHT_MARKER in content


def render_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool = False,
    response_part: str = RESPONSE_PART,
) -> str:
    """
    Render a completed conversation for training.

    `.removeprefix("<bos>")` matches the reference notebook: the tokenizer adds BOS
    again at tokenization time, and a doubled BOS silently degrades training.

    `enable_thinking` is documented for inference, so not every template version
    accepts it on a completed turn; fall back rather than fail, and let --dry_run show
    which path was taken.
    """
    # Targets containing a thought block must bypass the template, which would strip
    # it. Detected via the response_part marker rather than a hardcoded 31B token, so
    # this works for checkpoints with different thought delimiters (E2B/E4B).
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if assistant and has_thought_block(str(assistant.get("content", "")), response_part):
        return render_traced_example(tokenizer, messages, enable_thinking)

    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
    if enable_thinking:
        try:
            text = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=True)
            return text.removeprefix("<bos>")
        except TypeError:
            print(
                "  NOTE: this chat template does not accept enable_thinking on a "
                "completed turn; rendering without it.",
                flush=True,
            )

    text = tokenizer.apply_chat_template(messages, **kwargs)
    return text.removeprefix("<bos>")


def report_thought_channel(text: str, label: str) -> bool:
    """Whether a rendered assistant turn carries thought-channel markers."""
    has_thought = "channel>thought" in text or "<|channel>thought" in text
    print(f"  {label}: thought-channel markers present = {has_thought}")
    return has_thought


def probe_template_variants(model_name: str, messages: list[dict[str, str]]) -> None:
    """
    Does ANY template/flag combination put a thought block into a completed
    assistant turn?

    The claim being tested is that the thought channel is model-generated content
    rather than something the template inserts, so no flag can add it when the target
    text has none. Worth verifying rather than assuming, since the whole target-format
    decision rests on it.
    """
    from transformers import AutoTokenizer
    from unsloth.chat_templates import get_chat_template

    print("\n" + "=" * 78)
    print("TEMPLATE PROBE - can any flag inject a thought block?")
    print("=" * 78)

    variants = [
        ("gemma-4-thinking", {}),
        ("gemma-4-thinking", {"enable_thinking": True}),
        ("gemma-4", {}),
        ("gemma-4", {"enable_thinking": True}),
    ]

    for template_name, extra_kwargs in variants:
        label = f"{template_name} + {extra_kwargs or 'no flags'}"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer = get_chat_template(tokenizer, chat_template=template_name)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                **extra_kwargs,
            ).removeprefix("<bos>")
        except Exception as exc:
            print(f"  {label:48s} -> unavailable: {type(exc).__name__}: {exc}")
            continue

        has_thought = "channel>thought" in text
        tail = text[-90:].replace("\n", "\\n")
        print(f"  {label:48s} -> thought block: {has_thought}")
        print(f"  {'':48s}    tail: ...{tail}")

    print(
        "\n  If every row says False, the thought channel cannot come from the template.\n"
        "  It has to be present in the target text itself, or not at all."
    )

    # The decisive question for the suppression risk: at inference time, does
    # enable_thinking PRE-FILL the thought marker into the generation prompt?
    #
    # If it does, the model is already inside the thought channel before it emits a
    # single token, and finetuning on answer-only targets cannot stop it thinking.
    # If it does not, emitting <|channel>thought is the model's own choice, and
    # training against 297 examples that skip it can plausibly erode that choice.
    print("\n" + "=" * 78)
    print("GENERATION PROMPT PROBE - does enable_thinking pre-fill the thought marker?")
    print("=" * 78)

    user_only = [m for m in messages if m.get("role") == "user"]

    for template_name in ("gemma-4-thinking", "gemma-4"):
        for flag in (True, False):
            label = f"{template_name} + enable_thinking={flag}"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer = get_chat_template(tokenizer, chat_template=template_name)
                text = tokenizer.apply_chat_template(
                    user_only,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=flag,
                )
            except Exception as exc:
                print(f"  {label:44s} -> unavailable: {type(exc).__name__}")
                continue

            prefilled = "channel>thought" in text[-200:]
            print(f"  {label:44s} -> thought marker pre-filled: {prefilled}")
            print(f"  {'':44s}    prompt ends: ...{text[-70:]!r}")

    print(
        "\n  pre-filled True  -> the model cannot skip thinking; suppression risk is moot.\n"
        "  pre-filled False -> emitting the thought marker is the model's own choice,\n"
        "                      and answer-only finetuning could erode it. ft_03 decides."
    )


def do_dry_run(args: argparse.Namespace, repo_root: Path) -> None:
    """Tokenizer-only inspection. Runs on a laptop; no model weights are downloaded."""
    from transformers import AutoTokenizer
    from unsloth.chat_templates import get_chat_template

    print("=" * 78)
    print("DRY RUN - tokenizer only, no model loaded")
    print("=" * 78)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    train = load_jsonl(repo_root / args.train_path)
    example = train[0]
    text = render_example(
        tokenizer, example["messages"], args.enable_thinking_in_prompt, args.response_part
    )

    # Print the dataset path. --train_path defaults to the answer-only dataset, so
    # omitting it silently inspects the wrong data and the output still looks sane.
    print(f"\nDataset: {args.train_path}  ({len(train)} rows)")
    print(f"Example: {example['session_name']}")
    print(f"Chat template: {args.chat_template}")
    print(f"response_part: {args.response_part!r}")
    print(f"enable_thinking_in_prompt: {args.enable_thinking_in_prompt}")
    target_has_thought = has_thought_block(
        str(next((m for m in example["messages"] if m.get("role") == "assistant"), {}).get("content", "")),
        args.response_part,
    )
    print(
        f"Render path: {'manual (target has a thought block)' if target_has_thought else 'apply_chat_template'}"
    )
    print(f"Rendered length: {len(text)} chars")

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    print(f"Token count: {len(ids)}  (max_seq_length={args.max_seq_length})")
    if len(ids) > args.max_seq_length:
        print("  !! EXCEEDS max_seq_length - this example would be truncated")

    print("\n--- rendered tail (last 600 chars) ---")
    print(text[-600:])

    print("\n--- approximate loss span (text after the response marker) ---")
    # Must use the ARGUMENT, not the module default: the traced dataset anchors on
    # "<channel|>" so the thought stays unsupervised, and reporting the default
    # would show the thought inside the loss span and look like a masking failure.
    marker = args.response_part
    marker_index = text.find(marker)
    if marker_index == -1:
        print(f"  !! response marker {marker!r} NOT FOUND.")
        print("  train_on_responses_only would mask everything. Check the template name.")
    else:
        trained_span = text[marker_index + len(marker) :]
        print(repr(trained_span))
        print(f"\n  trained span: {len(trained_span)} chars")

    print("\n--- thought channel check ---")
    has_thought = report_thought_channel(text, "rendered assistant turn")

    # The traced design pairs a thought-anchored response_part with targets that
    # contain a thought block. Either one without the other is a misconfiguration:
    # the mask would land in the wrong place, or nothing would be supervised at all.
    anchored_on_thought = "channel" in args.response_part
    if anchored_on_thought and not has_thought:
        print(
            "\n  !! MISMATCH: --response_part is anchored on the thought channel but "
            "this example\n     has no thought block. You are almost certainly pointing "
            "at the answer-only\n     dataset - check the --train_path printed above."
        )
    elif has_thought and not anchored_on_thought:
        print(
            "\n  !! MISMATCH: this example HAS a thought block but --response_part is "
            "the default,\n     so the loss would cover the reasoning as well as the "
            "answer. Pass --response_part '<channel|>'."
        )
    print(
        "\nIf no thought markers appear above, the targets are answer-only and training\n"
        "will push the model to skip the thought channel. Decide that deliberately\n"
        "before running, since the voting eval uses thinking mode."
    )

    probe_template_variants(args.model_name, example["messages"])
    print("\nThis is an approximation of train_on_responses_only (string-level).")
    print("Run --inspect_only on the GPU node for the exact token-level mask.")


def build_model_and_tokenizer(args: argparse.Namespace):
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    # Shares the loader policy with the voting evaluation (see model_utils), so the
    # base being finetuned is loaded identically to the base being compared against.
    tokenizer, model = load_local_model_for_training(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    return model, tokenizer


def build_datasets(args: argparse.Namespace, repo_root: Path, tokenizer: Any):
    from datasets import Dataset

    def to_dataset(path: Path) -> Any:
        rows = load_jsonl(path)
        return Dataset.from_list(
            [
                {
                    "session_name": row["session_name"],
                    "text": render_example(
                        tokenizer,
                        row["messages"],
                        args.enable_thinking_in_prompt,
                        args.response_part,
                    ),
                }
                for row in rows
            ]
        )

    train = to_dataset(repo_root / args.train_path)
    val = to_dataset(repo_root / args.val_path)

    if args.limit_train_examples > 0:
        train = train.select(range(min(args.limit_train_examples, len(train))))
        val = val.select(range(min(args.limit_train_examples, len(val))))
        print(
            f"!! SMOKE TEST: truncated to {len(train)} train / {len(val)} val examples. "
            "The resulting adapter is not usable for the experiment.",
            flush=True,
        )

    return train, val


def print_exact_loss_span(trainer: Any, tokenizer: Any) -> None:
    """Decode what the collator actually supervises. This is the definitive check."""
    print("\n" + "=" * 78)
    print("EXACT LOSS SPAN (after train_on_responses_only)")
    print("=" * 78)

    batch = trainer.data_collator([trainer.train_dataset[0]])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    supervised = [int(t) for t, l in zip(input_ids, labels) if int(l) != -100]
    masked_count = int(sum(1 for l in labels if int(l) == -100))

    print(f"total tokens:      {len(input_ids)}")
    print(f"masked (-100):     {masked_count}")
    print(f"supervised tokens: {len(supervised)}")

    if not supervised:
        raise RuntimeError(
            "Nothing is supervised - train_on_responses_only masked the whole sequence. "
            f"Check that response_part={args.response_part!r} matches this chat template."
        )

    decoded = tokenizer.decode(supervised)
    print("\n--- supervised text ---")
    print(repr(decoded))

    if "roles" not in decoded:
        print(
            "\n  !! WARNING: the supervised span does not contain 'roles'. "
            "The mask is probably misaligned."
        )
    report_thought_channel(decoded, "supervised span")


def run_smoke_test(
    model: Any,
    model_io: Any,
    model_name: str,
    val_rows: list[dict[str, Any]],
    n: int,
    max_new_tokens: int = 10000,
) -> None:
    """
    Confirm the thought channel still fires after training.

    Generation goes through call_local_model, i.e. the exact path run_llm_votes uses.
    That matters twice over: Gemma 4's model_io is a multimodal Processor whose
    apply_chat_template rejects plain-string content when tokenize=True, and the
    `internal_thoughts` measured here is then the same field the voting results record
    (non-empty in all 573 games of the prompt_v4 baseline runs).
    """
    from unsloth import FastModel

    from src.utils.model_utils import call_local_model

    if n <= 0:
        return

    print("\n" + "=" * 78)
    print("POST-TRAINING SMOKE TEST - is the thought channel still alive?")
    print("=" * 78)

    try:
        FastModel.for_inference(model)
    except Exception as exc:
        print(f"  (for_inference unavailable: {exc}; falling back to model.eval())")
        model.eval()

    sampled = val_rows[:n]
    empty_thoughts = 0

    for row in sampled:
        try:
            text, debug_info = call_local_model(
                model=model,
                model_io=model_io,
                prompt=row["prompt"],
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                gemma_enable_thinking=True,
                return_debug_info=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        except Exception as exc:
            print(f"\n[{row['session_name']}] generation FAILED: {type(exc).__name__}: {exc}")
            continue

        debug_info = debug_info or {}
        thoughts = debug_info.get("internal_thoughts")
        parsed_thought = bool(thoughts and str(thoughts).strip())

        # A thought that runs past max_new_tokens never emits its closing marker, so
        # parse_reasoning_response cannot split it and returns the raw text instead.
        # That is truncation, NOT suppression - detect the opening marker directly so
        # the two are never confused.
        emitted_marker = "channel>thought" in (text or "")
        has_thought = parsed_thought or emitted_marker

        out_tokens = debug_info.get("output_token_count")
        truncated = bool(out_tokens and out_tokens >= max_new_tokens)

        if not has_thought:
            empty_thoughts += 1

        print(f"\n[{row['session_name']}] thought channel emitted: {has_thought}")
        print(f"  parsed cleanly: {parsed_thought}   output tokens: {out_tokens}/{max_new_tokens}")
        if truncated:
            print("  !! hit the token cap - raise --smoke_test_max_new_tokens")
        elif emitted_marker and not parsed_thought:
            print("  !! thought emitted but not parseable (no closing marker)")
        print(f"  gold:   {row['completion']}")
        print(f"  answer: {text[:200]!r}")
        if parsed_thought:
            print(f"  thought (first 200 chars): {str(thoughts)[:200]!r}")

    print(
        f"\n  thought channel emitted in {len(sampled) - empty_thoughts}/{len(sampled)} "
        "sampled generations (baseline: 573/573 across prompt_v4 runs 1-3)."
    )
    if empty_thoughts:
        print(
            "  !! Reasoning may have been suppressed by finetuning. Check an earlier "
            "epoch checkpoint before spending GPU hours on the voting reruns."
        )


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    if args.dry_run:
        do_dry_run(args, repo_root)
        return

    set_all_seeds(args.seed)

    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(args)
    train_dataset, val_dataset = build_datasets(args, repo_root, tokenizer)

    print(f"Model:      {args.model_name}")
    print(f"Train / val: {len(train_dataset)} / {len(val_dataset)}")
    print(f"LoRA:        r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"Epochs:      {args.num_train_epochs}  lr={args.learning_rate}")
    print(f"Output:      {output_dir}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.eval_strategy != "no" else None,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            report_to=args.report_to,
            save_strategy="epoch",
            eval_strategy=args.eval_strategy,
            # Only relevant when eval is on; keeps the fp32 logit cast survivable.
            per_device_eval_batch_size=1,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=args.response_part,
    )

    print_exact_loss_span(trainer, tokenizer)

    if args.inspect_only:
        print("\n--inspect_only set: stopping before training.")
        return

    stats = trainer.train()

    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    run_config = {
        "model_name": args.model_name,
        "chat_template": args.chat_template,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "num_train_examples": len(train_dataset),
        "num_val_examples": len(val_dataset),
        "targets_include_reasoning": False,
        "train_metrics": getattr(stats, "metrics", None),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nAdapter saved to {final_dir}")
    print(f"Per-epoch checkpoints under {output_dir}")

    run_smoke_test(
        model=model,
        model_io=tokenizer,
        model_name=args.model_name,
        val_rows=load_jsonl(repo_root / args.val_path),
        n=args.smoke_test_samples,
        max_new_tokens=args.smoke_test_max_new_tokens,
    )


if __name__ == "__main__":
    main()
