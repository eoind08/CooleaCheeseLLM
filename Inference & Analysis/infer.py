import torch
import tiktoken
import __main__

from model import (
    GPT,
    GPTConfig,
    Block,
    CausalSelfAttention,
    MLP,
)
__main__.GPT = GPT
__main__.GPTConfig = GPTConfig
__main__.Block = Block
__main__.CausalSelfAttention = CausalSelfAttention
__main__.MLP = MLP


MODEL_PATH = "../Results/Gruyere-v1.0-r1/log/model_19999_full.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 0.7
TOP_K = 40
MAX_NEW_TOKENS = 100


print(f"Loading model on {DEVICE}...")

model = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.eval()
model.to(DEVICE)

enc = tiktoken.get_encoding("gpt2")

print("Model loaded!")
print("Type 'exit' to quit.\n")

'''prompt = input("Prompt: ")

tokens = enc.encode(prompt)
x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

with torch.no_grad():
    logits, _ = model(x)

logits = logits[:, -1, :]
probs = torch.softmax(logits, dim=-1)

top_probs, top_tokens = torch.topk(probs, 20)

print("\nTop 20 next-token predictions:\n")

for prob, token in zip(top_probs[0], top_tokens[0]):
    token_id = token.item()
    token_text = enc.decode([token_id])
    print(f"{repr(token_text):20} {prob.item() * 100:8.4f}%")'''

while True:
    prompt = input("Prompt: ")

    if prompt.lower() == "exit":
        break

    tokens = enc.encode(prompt)

    tokens = tokens[-model.config.block_size:]

    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE,
    )

    prompt_length = x.size(1)

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            x_cond = x[:, -model.config.block_size:]

            logits, _ = model(x_cond)

            logits = logits[:, -1, :]
            if TEMPERATURE == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / TEMPERATURE

                if TOP_K is not None:
                    values, indices = torch.topk(logits, TOP_K)
                    probs = torch.softmax(values, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    next_token = indices.gather(-1, sampled)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token), dim=1)

    generated_tokens = x[0, prompt_length:].tolist()
    response = enc.decode(generated_tokens)

    print("\nResponse: " + response)
    print()