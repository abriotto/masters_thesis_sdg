from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from google import genai


ModelFamily = Literal["gemma4", "gpt_oss", "qwen", "standard_chat"]
ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelIOInfo:
    family: ModelFamily
    io_type: str
    has_chat_template: bool


@dataclass
class LocalModelBundle:
    model_name: str
    family: ModelFamily
    model_io: Any
    model: Any

    @property
    def io_info(self) -> dict:
        return {
            "family": self.family,
            "io_type": "processor" if self.family == "gemma4" else "tokenizer",
            "has_chat_template": has_chat_template(self.model_io),
        }


def call_gemini(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (response.text or "").strip()


def get_model_family(model_name: str) -> ModelFamily:
    low = model_name.lower()

    if "gemma-4" in low or "gemma4" in low:
        return "gemma4"

    if "gpt-oss" in low:
        return "gpt_oss"

    if "qwen" in low:
        return "qwen"

    return "standard_chat"


def has_chat_template(model_io: Any) -> bool:
    return (
        hasattr(model_io, "apply_chat_template")
        and getattr(model_io, "chat_template", None) is not None
    )


def get_model_io_info(model_name: str, model_io: Any) -> dict:
    family = get_model_family(model_name)

    return {
        "family": family,
        "io_type": "processor" if family == "gemma4" else "tokenizer",
        "has_chat_template": has_chat_template(model_io),
    }


def _validate_reasoning_effort(reasoning_effort: str) -> ReasoningEffort:
    valid_efforts = {"low", "medium", "high"}

    if reasoning_effort not in valid_efforts:
        raise ValueError(
            f"Unsupported reasoning effort: {reasoning_effort}. "
            f"Expected one of: {sorted(valid_efforts)}"
        )

    return reasoning_effort  # type: ignore[return-value]


def _load_torch_and_transformers():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def _best_model_dtype(torch_module):
    """
    Use bf16 on CUDA when supported, otherwise fp16.
    Fall back to fp32 on CPU.

    This avoids torch_dtype='auto', which can leave too little VRAM headroom
    for generation on large local models.
    """
    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16

    return torch_module.float32


def _model_load_kwargs(torch_module) -> dict:
    kwargs = {
        "torch_dtype": _best_model_dtype(torch_module),
    }

    if torch_module.cuda.is_available():
        kwargs["device_map"] = "auto"

    return kwargs


def _load_model_io(model_name: str, family: ModelFamily):
    _, _, AutoProcessor, AutoTokenizer = _load_torch_and_transformers()

    if family == "gemma4":
        return AutoProcessor.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_local_model(model_name: str) -> Tuple[Any, Any]:
    """
    Load a local HF causal LM and its tokenizer/processor.

    No quantization is used.
    CUDA models are loaded in bf16 when supported, otherwise fp16.
    """
    torch, AutoModelForCausalLM, _, _ = _load_torch_and_transformers()

    family = get_model_family(model_name)
    model_io = _load_model_io(model_name, family)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_model_load_kwargs(torch),
    )

    model.eval()
    return model_io, model


def load_local_model_bundle(model_name: str) -> LocalModelBundle:
    model_io, model = load_local_model(model_name=model_name)
    family = get_model_family(model_name)

    return LocalModelBundle(
        model_name=model_name,
        family=family,
        model_io=model_io,
        model=model,
    )


class BaseLocalModelHandler:
    """
    Generic local HF model handler.

    Policy:
    - If a chat template exists, always use it.
    - If no chat template exists, use raw prompt only for generic/base models.
    - Specific families can require chat templates when raw fallback is unsafe.
    """

    requires_chat_template = False

    def __init__(
        self,
        model: Any,
        model_io: Any,
        model_name: Optional[str],
        reasoning_effort: str = "low",
        gemma_enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.model_io = model_io
        self.model_name = model_name or ""
        self.family = get_model_family(self.model_name)
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
        messages = self.build_messages(prompt)

        if has_chat_template(self.model_io):
            return self.apply_chat_template(messages)

        if self.requires_chat_template:
            raise ValueError(
                f"Model '{self.model_name}' requires a chat template, "
                "but none was found."
            )

        return prompt

    def tokenize(self, input_text: str, model_device: Any):
        inputs = self.model_io(
            text=input_text,
            return_tensors="pt",
            truncation=False,
        )
        return inputs.to(model_device)

    def decode(self, generated_ids: Any) -> str:
        return self.model_io.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def pad_token_id(self) -> Optional[int]:
        pad_token_id = getattr(self.model_io, "pad_token_id", None)

        if pad_token_id is None:
            pad_token_id = getattr(self.model_io, "eos_token_id", None)

        return pad_token_id


class StandardChatHandler(BaseLocalModelHandler):
    pass


class QwenHandler(BaseLocalModelHandler):
    requires_chat_template = True


class GptOssHandler(BaseLocalModelHandler):
    """
    GPT-OSS must use the Transformers chat template.

    The template applies the Harmony response format automatically.
    reasoning_effort is passed to the template when supported by the installed
    Transformers/tokenizer version.
    """

    requires_chat_template = True

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=self.reasoning_effort,
            )
        except TypeError:
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


class Gemma4Handler(BaseLocalModelHandler):
    """
    Gemma 4 uses an AutoProcessor.

    Thinking is disabled by default, but can be enabled through
    gemma_enable_thinking=True.
    """

    requires_chat_template = True

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.gemma_enable_thinking,
            )
        except TypeError:
            return self.model_io.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def decode(self, generated_ids: Any) -> str:
        raw = self.model_io.decode(
            generated_ids,
            skip_special_tokens=False,
        )

        parse_response = getattr(self.model_io, "parse_response", None)

        if callable(parse_response):
            parsed = parse_response(raw)

            if isinstance(parsed, str):
                return parsed.strip()

            if isinstance(parsed, dict):
                content = parsed.get("content")

                if isinstance(content, str):
                    return content.strip()

                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                    if parts:
                        return "".join(parts).strip()

            return str(parsed).strip()

        return raw.strip()


def make_local_handler(
    model: Any,
    model_io: Any,
    model_name: Optional[str],
    reasoning_effort: str = "low",
    gemma_enable_thinking: bool = False,
) -> BaseLocalModelHandler:
    family = get_model_family(model_name or "")

    if family == "gemma4":
        return Gemma4Handler(
            model=model,
            model_io=model_io,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            gemma_enable_thinking=gemma_enable_thinking,
        )

    if family == "gpt_oss":
        return GptOssHandler(
            model=model,
            model_io=model_io,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            gemma_enable_thinking=False,
        )

    if family == "qwen":
        return QwenHandler(
            model=model,
            model_io=model_io,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            gemma_enable_thinking=False,
        )

    return StandardChatHandler(
        model=model,
        model_io=model_io,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        gemma_enable_thinking=False,
    )


def call_local_model(
    model: Any,
    model_io: Any,
    prompt: str,
    model_name: str | None = None,
    max_new_tokens: int = 768,
    reasoning_effort: str = "low",
    gemma_enable_thinking: bool = False,
    return_debug_info: bool = False,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 0,
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

    model_device = next(model.parameters()).device
    inputs = handler.tokenize(input_text, model_device)

    input_len = int(inputs["input_ids"].shape[-1])
    input_char_count = len(input_text)
    pad_token_id = handler.pad_token_id()

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "repetition_penalty": repetition_penalty,
    }

    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    t0 = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    t1 = time.time()

    generated_ids = outputs[0][input_len:]
    output_token_count = int(generated_ids.shape[0])
    text = handler.decode(generated_ids)

    family = get_model_family(model_name or "")

    debug_info = None
    if return_debug_info:
        debug_info = {
            "model_device": str(model_device),
            "input_token_count": input_len,
            "output_token_count": output_token_count,
            "generation_time_sec": float(t1 - t0),
            "used_chat_template": has_chat_template(model_io),
            "input_char_count": input_char_count,
            "model_family": family,
            "handler": handler.__class__.__name__,
            "reasoning_effort": (
                reasoning_effort
                if family == "gpt_oss"
                else None
            ),
            "gemma_enable_thinking": (
                bool(gemma_enable_thinking)
                if family == "gemma4"
                else None
            ),
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "torch_dtype": str(getattr(model, "dtype", "unknown")),
        }

    del outputs
    del inputs
    del generated_ids

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_debug_info:
        return text, debug_info

    return text