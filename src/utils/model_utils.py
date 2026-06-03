from __future__ import annotations

import unsloth

import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple


ModelFamily = Literal["gemma4", "gpt_oss", "qwen"]
ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelIOInfo:
    family: ModelFamily
    io_type: str
    has_chat_template: bool
    backend: str = "unsloth"


@dataclass(frozen=True)
class LoaderPolicy:
    loaders: tuple[str, ...]
    load_in_4bit: bool = True
    attn_implementation: Optional[str] = None
    quantization_label: str = "unsloth_load_in_4bit"


@dataclass
class LocalModelBundle:
    model_name: str
    family: ModelFamily
    model_io: Any
    model: Any
    backend: str = "unsloth"
    max_seq_length: Optional[int] = None

    @property
    def io_info(self) -> dict:
        return {
            "family": self.family,
            "io_type": "tokenizer_or_processor",
            "has_chat_template": has_chat_template(self.model_io),
            "backend": self.backend,
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
        }


# ---------------------------------------------------------------------------
# Model family / loader policy
# ---------------------------------------------------------------------------

def get_model_family(model_name: str) -> ModelFamily:
    """
    Local Unsloth model families supported by this project.

    Current final experiment:
    - GPT-OSS through Unsloth FastLanguageModel.
    - Gemma 4 through Unsloth FastModel.

    Future extension:
    - Qwen through Unsloth FastLanguageModel.
    """
    low = model_name.lower()

    if "gemma-4" in low or "gemma4" in low:
        return "gemma4"

    if "gpt-oss" in low or "gpt_oss" in low:
        return "gpt_oss"

    if "qwen" in low:
        return "qwen"

    raise ValueError(
        f"Unsupported local model family for Unsloth runner: {model_name!r}. "
        "Expected a Gemma 4, GPT-OSS, or Qwen Unsloth checkpoint."
    )


def get_loader_policy(model_name: str, family: ModelFamily) -> LoaderPolicy:
    """
    Centralized model-loading policy.

    Notes from the current experiments:
    - GPT-OSS needs FastLanguageModel first and `attn_implementation='eager'`
      in this cluster stack, otherwise Transformers tries unsupported SDPA.
    - GPT-OSS BNB checkpoint still needs `load_in_4bit=True` here; setting it
      to False pushed Unsloth into a huge 16-bit LoRA path.
    - Gemma 4 prefers FastModel.
    - Qwen support is intentionally simple and future-facing.
    """
    is_bnb_checkpoint = "bnb-4bit" in model_name.lower()
    quantization_label = (
        "unsloth_load_in_4bit_on_bnb_checkpoint"
        if is_bnb_checkpoint
        else "unsloth_load_in_4bit"
    )

    if family == "gpt_oss":
        return LoaderPolicy(
            loaders=("FastLanguageModel", "FastModel"),
            load_in_4bit=True,
            attn_implementation="eager",
            quantization_label=quantization_label,
        )

    if family == "gemma4":
        return LoaderPolicy(
            loaders=("FastModel", "FastLanguageModel"),
            load_in_4bit=True,
            attn_implementation=None,
            quantization_label=quantization_label,
        )

    if family == "qwen":
        return LoaderPolicy(
            loaders=("FastLanguageModel", "FastModel"),
            load_in_4bit=True,
            attn_implementation=None,
            quantization_label=quantization_label,
        )

    raise AssertionError(f"Unhandled model family: {family}")


def has_chat_template(model_io: Any) -> bool:
    return (
        hasattr(model_io, "apply_chat_template")
        and getattr(model_io, "chat_template", None) is not None
    )


def get_model_io_info(model_name: str, model_io: Any) -> dict:
    family = get_model_family(model_name)
    return {
        "family": family,
        "io_type": "tokenizer_or_processor",
        "has_chat_template": has_chat_template(model_io),
        "backend": "unsloth",
    }


def _validate_reasoning_effort(reasoning_effort: str) -> ReasoningEffort:
    valid_efforts = {"low", "medium", "high"}
    if reasoning_effort not in valid_efforts:
        raise ValueError(
            f"Unsupported reasoning effort: {reasoning_effort}. "
            f"Expected one of: {sorted(valid_efforts)}"
        )
    return reasoning_effort  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Unsloth loading
# ---------------------------------------------------------------------------

def _load_unsloth_classes():
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required for local models. "
            "Install/update it in the environment where you run the experiments."
        ) from exc

    try:
        from unsloth import FastModel
    except Exception:
        FastModel = None

    return FastLanguageModel, FastModel


def _loader_registry(FastLanguageModel: Any, FastModel: Any) -> dict[str, Any]:
    registry = {"FastLanguageModel": FastLanguageModel}
    if FastModel is not None:
        registry["FastModel"] = FastModel
    return registry


def _safe_set_tokenizer_defaults(tokenizer_or_processor: Any) -> None:
    if getattr(tokenizer_or_processor, "pad_token", None) is None:
        eos = getattr(tokenizer_or_processor, "eos_token", None)
        if eos is not None:
            tokenizer_or_processor.pad_token = eos

    if hasattr(tokenizer_or_processor, "padding_side"):
        tokenizer_or_processor.padding_side = "left"


def _unsloth_from_pretrained(
    model_name: str,
    family: ModelFamily,
    max_seq_length: int,
    dtype: Any = None,
    full_finetuning: bool = False,
) -> Tuple[Any, Any, str]:
    """
    Unsloth-only loader.

    No Gemini dependency.
    No legacy HF-local fallback.
    No checkpoint aliasing/normalization.
    """
    FastLanguageModel, FastModel = _load_unsloth_classes()
    registry = _loader_registry(FastLanguageModel, FastModel)
    policy = get_loader_policy(model_name, family)

    common_kwargs = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": policy.load_in_4bit,
        "full_finetuning": full_finetuning,
    }

    if policy.attn_implementation is not None:
        common_kwargs["attn_implementation"] = policy.attn_implementation

    errors: list[str] = []
    for loader_name in policy.loaders:
        loader = registry.get(loader_name)
        if loader is None:
            errors.append(f"{loader_name}: unavailable in this Unsloth installation")
            continue

        try:
            model, model_io = loader.from_pretrained(**common_kwargs)

            # Some Unsloth helpers mutate the model in-place and return None.
            for_inference = getattr(loader, "for_inference", None)
            if callable(for_inference):
                maybe_model = for_inference(model)
                if maybe_model is not None:
                    model = maybe_model

            if model is None:
                raise RuntimeError(
                    f"{loader_name}.from_pretrained returned model=None for {model_name}"
                )

            if model_io is None:
                raise RuntimeError(
                    f"{loader_name}.from_pretrained returned tokenizer/processor=None "
                    f"for {model_name}"
                )

            setattr(model, "_local_backend", "unsloth")
            setattr(model, "_local_loader", loader_name)
            setattr(model, "_local_quantization", policy.quantization_label)
            setattr(model, "_requested_model_name", model_name)
            setattr(model, "_resolved_model_name", model_name)
            setattr(model, "_max_seq_length", max_seq_length)
            setattr(model, "_last_thought_block", None)
            setattr(model, "_last_prompt_formatter", None)

            _safe_set_tokenizer_defaults(model_io)
            model.eval()
            return model_io, model, loader_name

        except Exception as exc:
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not load model with Unsloth: {model_name}\n"
        f"Tried loaders:\n" + "\n".join(errors)
    )


def load_local_model(
    model_name: str,
    max_seq_length: int = 8192,
    dtype: Any = None,
    full_finetuning: bool = False,
) -> Tuple[Any, Any]:
    family = get_model_family(model_name)
    model_io, model, _ = _unsloth_from_pretrained(
        model_name=model_name,
        family=family,
        max_seq_length=max_seq_length,
        dtype=dtype,
        full_finetuning=full_finetuning,
    )
    return model_io, model


def load_local_model_bundle(
    model_name: str,
    max_seq_length: int = 8192,
    dtype: Any = None,
    full_finetuning: bool = False,
) -> LocalModelBundle:
    family = get_model_family(model_name)
    model_io, model, _ = _unsloth_from_pretrained(
        model_name=model_name,
        family=family,
        max_seq_length=max_seq_length,
        dtype=dtype,
        full_finetuning=full_finetuning,
    )
    return LocalModelBundle(
        model_name=model_name,
        family=family,
        model_io=model_io,
        model=model,
        max_seq_length=max_seq_length,
    )


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def _normalize_device_value(device_value: Any):
    if device_value is None:
        return None

    value = str(device_value)
    if value in {"cpu", "disk", "meta"}:
        return None

    if isinstance(device_value, int) or value.isdigit():
        return f"cuda:{value}"

    return value


def get_model_input_device(model: Any):
    """
    Pick the right input device for Unsloth/Accelerate models.
    """
    import torch

    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        return torch.device(model_device)

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        preferred_keys = (
            "model.embed_tokens",
            "transformer.wte",
            "model.model.embed_tokens",
            "language_model.model.embed_tokens",
            "",
        )

        for key in preferred_keys:
            if key in hf_device_map:
                normalized = _normalize_device_value(hf_device_map[key])
                if normalized is not None:
                    return torch.device(normalized)

        for value in hf_device_map.values():
            normalized = _normalize_device_value(value)
            if normalized is not None:
                return torch.device(normalized)

    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _strip_known_special_tokens(text: str) -> str:
    replacements = [
        "<bos>",
        "<eos>",
        "<pad>",
        "<turn|>",
        "<|end|>",
        "<|return|>",
    ]
    for token in replacements:
        text = text.replace(token, "")

    text = re.sub(r"<\|turn\>(?:assistant|model)\s*", "", text)
    text = re.sub(r"<\|turn\>(?:user|system)\s*", "", text)
    text = re.sub(r"<\|start\|>assistant", "", text)
    return text.strip()


def _parse_gemma_thinking(raw: str) -> tuple[Optional[str], Optional[str]]:
    match = re.search(
        r"<\|channel\>thought\s*(?P<thinking>.*?)(?:<channel\|>)(?P<answer>.*)$",
        raw,
        flags=re.DOTALL,
    )
    if not match:
        return None, None

    thinking = _strip_known_special_tokens(match.group("thinking"))
    answer = _strip_known_special_tokens(match.group("answer"))
    return thinking or None, answer or None


def _parse_harmony_response(raw: str) -> tuple[Optional[str], Optional[str]]:
    analysis_blocks = re.findall(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>assistant|<\|channel\|>final|$)",
        raw,
        flags=re.DOTALL,
    )
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>|$)",
        raw,
        flags=re.DOTALL,
    )

    thinking = "\n\n".join(_strip_known_special_tokens(x) for x in analysis_blocks).strip()
    answer = _strip_known_special_tokens(final_match.group(1)) if final_match else None
    return thinking or None, answer or None


def parse_reasoning_response(model_io: Any, raw: str) -> tuple[str, Optional[str]]:
    """
    Return (visible_answer, internal_thoughts).

    Supports:
    - tokenizer/processor.parse_response if available;
    - Gemma thinking tags;
    - GPT-OSS Harmony tags;
    - plain text fallback.
    """
    parse_response = getattr(model_io, "parse_response", None)
    if callable(parse_response):
        try:
            parsed = parse_response(raw)
            if isinstance(parsed, dict):
                thinking = parsed.get("thinking", None)
                content = parsed.get("content", parsed.get("answer", None))

                if isinstance(content, str):
                    return content.strip(), thinking

                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                    if parts:
                        return "".join(parts).strip(), thinking

                return str(parsed).strip(), thinking
        except Exception as exc:
            warnings.warn(f"parse_response failed; falling back to regex parser: {exc}")

    thinking, answer = _parse_gemma_thinking(raw)
    if answer is not None:
        return answer.strip(), thinking

    thinking, answer = _parse_harmony_response(raw)
    if answer is not None:
        return answer.strip(), thinking

    return _strip_known_special_tokens(raw), None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

class BaseLocalModelHandler:
    def __init__(
        self,
        model: Any,
        model_io: Any,
        model_name: str,
        reasoning_effort: str = "low",
        gemma_enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.model_io = model_io
        self.model_name = model_name
        self.family = get_model_family(model_name)
        self.reasoning_effort = _validate_reasoning_effort(reasoning_effort)
        self.gemma_enable_thinking = gemma_enable_thinking

    def build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self.model_io.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_input_text(self, prompt: str) -> str:
        if not has_chat_template(self.model_io):
            raise ValueError(
                f"Model '{self.model_name}' requires a chat template, but none was found."
            )
        return self.apply_chat_template(self.build_messages(prompt))

    def tokenize(self, input_text: str, model_device: Any):
        # `text=` is required for Gemma 4 processors; it also works for normal tokenizers.
        inputs = self.model_io(
            text=input_text,
            return_tensors="pt",
            truncation=False,
        )

        if inputs is None:
            raise RuntimeError(
                "model_io returned None during tokenization. "
                "Check that the processor/tokenizer supports text-only inputs."
            )

        return inputs.to(model_device)

    def decode(self, generated_ids: Any) -> str:
        raw = self.model_io.decode(generated_ids, skip_special_tokens=False)
        text, thoughts = parse_reasoning_response(self.model_io, raw)
        setattr(self.model, "_last_thought_block", thoughts)
        return text.strip()

    def pad_token_id(self) -> Optional[int]:
        pad_token_id = getattr(self.model_io, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.model_io, "eos_token_id", None)
        return pad_token_id


class GptOssHandler(BaseLocalModelHandler):
    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """
        GPT-OSS: use tokenizer chat template with reasoning_effort.
        This mirrors the working Unsloth notebook-style path.
        """
        setattr(self.model, "_last_prompt_formatter", "chat_template")
        try:
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=self.reasoning_effort,
            )
        except TypeError:
            warnings.warn(
                "tokenizer.apply_chat_template does not accept reasoning_effort; "
                "falling back without it."
            )
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


class Gemma4Handler(BaseLocalModelHandler):
    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        setattr(self.model, "_last_prompt_formatter", "chat_template")
        try:
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.gemma_enable_thinking,
            )
        except TypeError:
            if self.gemma_enable_thinking:
                messages = [{"role": "system", "content": "<|think|>"}, *messages]
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


class QwenHandler(BaseLocalModelHandler):
    """
    Future-facing Qwen support through Unsloth.

    This intentionally stays generic. Add model-specific options only after a
    dedicated Qwen smoke test confirms the exact expected chat template behavior.
    """
    pass


def make_local_handler(
    model: Any,
    model_io: Any,
    model_name: str,
    reasoning_effort: str = "low",
    gemma_enable_thinking: bool = False,
) -> BaseLocalModelHandler:
    family = get_model_family(model_name)

    if family == "gpt_oss":
        return GptOssHandler(model, model_io, model_name, reasoning_effort, False)

    if family == "gemma4":
        return Gemma4Handler(model, model_io, model_name, reasoning_effort, gemma_enable_thinking)

    if family == "qwen":
        return QwenHandler(model, model_io, model_name, reasoning_effort, False)

    raise AssertionError(f"Unhandled model family: {family}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def call_local_model(
    model: Any,
    model_io: Any,
    prompt: str,
    model_name: str,
    max_new_tokens: int = 768,
    reasoning_effort: str = "low",
    gemma_enable_thinking: bool = False,
    return_debug_info: bool = False,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 0,
    temperature: Optional[float] = None,
    top_p: float = 0.95,
    top_k: int = 64,
):
    import torch

    reasoning_effort = _validate_reasoning_effort(reasoning_effort)

    handler = make_local_handler(
        model=model,
        model_io=model_io,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        gemma_enable_thinking=gemma_enable_thinking,
    )

    input_text = handler.build_input_text(prompt)
    model_device = get_model_input_device(model)
    inputs = handler.tokenize(input_text, model_device)

    input_len = int(inputs["input_ids"].shape[-1])
    input_char_count = len(input_text)
    pad_token_id = handler.pad_token_id()

    do_sample = temperature is not None and temperature > 0

    effective_temperature = float(temperature) if do_sample else None
    effective_top_p = float(top_p) if do_sample else None
    effective_top_k = int(top_k) if do_sample else None

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
    }

    if do_sample:
        generation_kwargs["temperature"] = effective_temperature
        generation_kwargs["top_p"] = effective_top_p
        generation_kwargs["top_k"] = effective_top_k

    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    setattr(model, "_last_thought_block", None)

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    t1 = time.time()

    if outputs is None:
        raise RuntimeError(
            "model.generate returned None. This usually means generation failed internally "
            "or the loaded model/backend does not expose the standard Transformers generate API."
        )

    generated_ids = outputs[0][input_len:]
    output_token_count = int(generated_ids.shape[0])
    text = handler.decode(generated_ids)

    family = get_model_family(model_name)

    debug_info = None
    if return_debug_info:
        debug_info = {
            "backend": getattr(model, "_local_backend", "unsloth"),
            "loader": getattr(model, "_local_loader", None),
            "model_device": str(model_device),
            "input_token_count": input_len,
            "output_token_count": output_token_count,
            "generation_time_sec": float(t1 - t0),
            "used_chat_template": has_chat_template(model_io),
            "input_char_count": input_char_count,
            "model_family": family,
            "handler": handler.__class__.__name__,
            "prompt_formatter": getattr(model, "_last_prompt_formatter", None),
            "reasoning_effort": reasoning_effort if family == "gpt_oss" else None,
            "gemma_enable_thinking": (
                bool(gemma_enable_thinking) if family == "gemma4" else None
            ),
            "do_sample": do_sample,
            "temperature": effective_temperature,
            "top_p": effective_top_p,
            "top_k": effective_top_k,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "torch_dtype": str(getattr(model, "dtype", "unknown")),
            "quantization": getattr(model, "_local_quantization", "unknown"),
            "requested_model_name": getattr(model, "_requested_model_name", model_name),
            "resolved_model_name": getattr(model, "_resolved_model_name", model_name),
            "max_seq_length": getattr(model, "_max_seq_length", None),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "internal_thoughts": getattr(model, "_last_thought_block", None),
        }

    del outputs
    del inputs
    del generated_ids

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_debug_info:
        return text, debug_info

    return text


__all__ = [
    "ModelFamily",
    "ReasoningEffort",
    "ModelIOInfo",
    "LocalModelBundle",
    "get_model_family",
    "get_model_io_info",
    "get_model_input_device",
    "load_local_model",
    "load_local_model_bundle",
    "call_local_model",
]
