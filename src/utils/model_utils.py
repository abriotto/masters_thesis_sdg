from __future__ import annotations

from typing import Tuple

from google import genai


def call_gemini(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (response.text or "").strip()


def _select_torch_dtype():
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_model_family(model_name: str) -> str:
    low = model_name.lower()

    if "gemma-4" in low:
        return "gemma4"
    if "gpt-oss" in low:
        return "gpt_oss"
    return "standard_chat"


def has_chat_template(model_io) -> bool:
    return hasattr(model_io, "apply_chat_template") and model_io.chat_template is not None


def get_model_io_info(model_name: str, model_io) -> dict:
    family = get_model_family(model_name)
    return {
        "family": family,
        "io_type": "processor" if family == "gemma4" else "tokenizer",
        "has_chat_template": has_chat_template(model_io),
    }


def load_local_model(model_name: str) -> Tuple[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dtype = _select_torch_dtype()
    family = get_model_family(model_name)

    if family == "gemma4":
        model_io = AutoProcessor.from_pretrained(model_name)
    else:
        model_io = AutoTokenizer.from_pretrained(model_name)
        if getattr(model_io, "pad_token", None) is None:
            model_io.pad_token = model_io.eos_token
        model_io.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    return model_io, model


def build_model_input_text(
    model_io,
    prompt: str,
    model_name: str | None = None,
    require_chat_template_for_gpt_oss: bool = True,
) -> str:
    family = get_model_family(model_name or "")
    messages = [{"role": "user", "content": prompt}]

    if has_chat_template(model_io):
        if family == "gemma4":
            try:
                return model_io.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return model_io.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        return model_io.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if require_chat_template_for_gpt_oss and family == "gpt_oss":
        raise ValueError(
            f"Model '{model_name}' appears to be a gpt-oss model, but no chat template was found."
        )

    return prompt


def _tokenize_inputs(model_io, input_text: str, model_device):
    inputs = model_io(
        text=input_text,
        return_tensors="pt",
        truncation=True,
    )
    return inputs.to(model_device)


def _decode_generated_text(model_io, generated_ids):
    return model_io.decode(generated_ids, skip_special_tokens=True).strip()


def call_local_model(
    model,
    model_io,
    prompt: str,
    model_name: str | None = None,
    max_new_tokens: int = 256,
    require_chat_template_for_gpt_oss: bool = True,
    return_debug_info: bool = False,
):
    import time
    import torch

    input_text = build_model_input_text(
        model_io=model_io,
        prompt=prompt,
        model_name=model_name,
        require_chat_template_for_gpt_oss=require_chat_template_for_gpt_oss,
    )

    model_device = next(model.parameters()).device
    inputs = _tokenize_inputs(model_io, input_text, model_device)

    input_len = inputs["input_ids"].shape[1]

    pad_token_id = getattr(model_io, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model_io, "eos_token_id", None)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    t1 = time.time()

    generated_ids = outputs[0][input_len:]
    text = _decode_generated_text(model_io, generated_ids)

    if not return_debug_info:
        return text

    debug_info = {
        "model_device": str(model_device),
        "input_token_count": int(input_len),
        "output_token_count": int(generated_ids.shape[0]),
        "generation_time_sec": float(t1 - t0),
        "used_chat_template": has_chat_template(model_io),
        "input_char_count": len(input_text),
        "model_family": get_model_family(model_name or ""),
    }
    return text, debug_info