from unsloth import FastLanguageModel
import torch

model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

print("CUDA available:", torch.cuda.is_available(), flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Torch:", torch.__version__, flush=True)
print("CUDA:", torch.version.cuda, flush=True)

print("Loading model...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,          # checkpoint is already BNB 4-bit
    full_finetuning=False,
    attn_implementation="eager" # needed in your env to avoid SDPA load error
)
print("Model loaded.", flush=True)

messages = [
    {
        "role": "user",
        "content": (
            "Return valid JSON only with exactly these keys: "
            "chosen_vote and justification. "
            "Use chosen_vote='Justin' and a one-sentence justification."
        ),
    }
]

print("Applying chat template...", flush=True)
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="high",
).to("cuda")

print("Input tokens:", inputs["input_ids"].shape[-1], flush=True)

print("Generating...", flush=True)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
    )
print("Generation done.", flush=True)

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=False), flush=True)