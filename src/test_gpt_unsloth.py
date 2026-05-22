from unsloth import FastModel
import torch

model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
)

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

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="high",
).to("cuda")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=False))