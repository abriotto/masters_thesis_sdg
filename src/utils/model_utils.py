from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from google import genai


ModelFamily = Literal["gemma4", "gpt_oss", "standard_chat"]
QuantizationMode = Literal["none", "8bit", "4bit"]
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
    quantization: QuantizationMode = "none"

    @property
    def io_info(self) -> dict:
        return {
            "family": self.family,
            "io_type": "processor" if self.family == "gemma4" else "tokenizer",
            "has_chat_template": has_chat_template(self.model_io),
            "quantization": self.quantization,
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


def _validate_quantization(quantization: str) -> QuantizationMode:
    valid_modes = {"none", "8bit", "4bit"}

    if quantization not in valid_modes:
        raise ValueError(
            f"Unsupported quantization mode: {quantization}. "
            f"Expected one of: {sorted(valid_modes)}"
        )

    return quantization  # type: ignore[return-value]


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


def _model_load_kwargs(torch_module, quantization: str = "none") -> dict:
    """
    Centralized loading policy.

    quantization='none':
        Loads the model with torch_dtype='auto'.

    quantization='8bit' or '4bit':
        Uses bitsandbytes through Transformers BitsAndBytesConfig.
        This requires CUDA and bitsandbytes installed.
    """
    quantization = _validate_quantization(quantization)

    if quantization == "none":
        kwargs = {
            "torch_dtype": "auto",
        }

        if torch_module.cuda.is_available():
            kwargs["device_map"] = "auto"

        return kwargs

    if not torch_module.cuda.is_available():
        raise RuntimeError(
            f"Quantization mode '{quantization}' requires CUDA, but CUDA is not available."
        )

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "8-bit/4-bit quantization requires bitsandbytes support through Transformers. "
            "Install the needed packages, for example: pip install bitsandbytes"
        ) from e

    kwargs = {
        "device_map": "auto",
    }

    if quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        return kwargs

    if quantization == "4bit":
        compute_dtype = (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return kwargs

    raise ValueError(f"Unsupported quantization mode: {quantization}")


def _load_model_io(model_name: str, family: ModelFamily):
    _, _, AutoProcessor, AutoTokenizer = _load_torch_and_transformers()

    if family == "gemma4":
        return AutoProcessor.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_local_model(
    model_name: str,
    quantization: str = "none",
) -> Tuple[Any, Any]:
    """
    Backward-compatible loader.

    The runner can keep using:

        model_io, model = load_local_model(model_name, quantization="none")
    """
    quantization = _validate_quantization(quantization)

    torch, AutoModelForCausalLM, _, _ = _load_torch_and_transformers()

    family = get_model_family(model_name)
    model_io = _load_model_io(model_name, family)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_model_load_kwargs(torch, quantization=quantization),
    )

    model.eval()
    return model_io, model


def load_local_model_bundle(
    model_name: str,
    quantization: str = "none",
) -> LocalModelBundle:
    """
    Optional cleaner API for future runners.
    """
    quantization = _validate_quantization(quantization)

    model_io, model = load_local_model(
        model_name=model_name,
        quantization=quantization,
    )
    family = get_model_family(model_name)

    return LocalModelBundle(
        model_name=model_name,
        family=family,
        model_io=model_io,
        model=model,
        quantization=quantization,
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
    """
    Default handler for Qwen, Llama, Mistral, and most instruct/chat models.

    If their tokenizer provides a chat template, it is used automatically.
    If not, the raw prompt is used as fallback.
    """

    pass


class GptOssHandler(BaseLocalModelHandler):
    """
    GPT-OSS must use the Transformers chat template.

    The template applies the Harmony response format automatically.
    reasoning_effort defaults to 'low' and is passed to the template when
    supported by the installed Transformers/tokenizer version.
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
    max_new_tokens: int = 256,
    reasoning_effort: str = "low",
    gemma_enable_thinking: bool = False,
    return_debug_info: bool = False,
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

    input_len = inputs["input_ids"].shape[-1]
    pad_token_id = handler.pad_token_id()

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }

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
    text = handler.decode(generated_ids)

    if not return_debug_info:
        return text

    family = get_model_family(model_name or "")

    debug_info = {
        "model_device": str(model_device),
        "input_token_count": int(input_len),
        "output_token_count": int(generated_ids.shape[0]),
        "generation_time_sec": float(t1 - t0),
        "used_chat_template": has_chat_template(model_io),
        "input_char_count": len(input_text),
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
    }

    return text, debug_info