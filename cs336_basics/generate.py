import torch
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.configs import RUNS, RunConfig
from cs336_basics.modules import TransformerLM, softmax
from cs336_basics.nn_utils import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import build_model, build_optimizer

# Must match scripts/tokenize_and_encode.py -- the tokenizer has to be rebuilt
# with the same special tokens it was trained with.
SPECIAL_TOKENS = ["<|endoftext|>"]

# vocab_to_id is keyed by bytes, not str.
END_OF_TEXT = SPECIAL_TOKENS[0].encode("utf-8")

# Sampling settings
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 256

# Menu order for the interactive picker: (RUNS key, description).
CHAT_RUNS = [
    ("h100", "Chat-small,  22M parameters, 327M tokens of the tinystories dataset."),
    ("owt", "Chat-small,  22M parameters, 327M tokens of the openwebtext dataset."),
    ("medium", "Chat-medium, 134M parameters, 655M tokens of the openwebtext dataset."),
]


def top_p_sampling(
    probs: Float[Tensor, "batch vocab"],
    top_p: float
) -> Float[Tensor, "batch vocab"]:
    """Top-p keeps only the most likely tokens, drops the rest, and rescales what
    remains to sum to 1.

    Args:
        probs: Probability rows of shape (batch, vocab) -- already normalized,
            so this runs after the softmax, not before. Modified in place.
            Each row gets its own nucleus.
        top_p (float): Cumulative probability threshold in (0, 1].

    Returns:
        The same tensor, with out-of-nucleus entries zeroed and the survivors
        renormalized to sum to 1.
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    keep_sorted = (cumsum - sorted_probs) < top_p
    keep_sorted[..., 0] = True

    keep = torch.zeros_like(keep_sorted).scatter(dim=-1, index=sorted_idx, src=keep_sorted)

    probs.masked_fill_(~keep, 0.0)
    probs /= probs.sum(dim=-1, keepdim=True)

    return probs

@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt: Int[Tensor, "1 seq_len"],
    max_context_len: int,
    max_tokens: int,
    end_token_id: int,
    temperature: float = 1,
    top_p: float = 1,
) -> Int[Tensor, "1 seq_len"]:
    """Sample a continuation of `prompt`, one token at a time.

    Each iteration is a full forward pass over the sequence so far, of which
    only the final position is used.

    Temperature divides the logits before the softmax. Below 1 it sharpens the
    distribution toward the argmax, above 1 it flattens toward uniform.
    
    Args:
        model: Maps (batch, seq) token IDs to (batch, seq, vocab) logits.
        prompt: Starting token IDs, shape (1, seq_len). Returned as a prefix of
            the result, so the caller gets prompt and completion together.
        max_context_len (int): The model's context window. Only the trailing
            window is fed forward; the returned sequence is never truncated.
        max_tokens (int): Cap on generated tokens. Generation may stop earlier
            if `end_token_id` is sampled.
        end_token_id (int): Sampling this ID stops generation. It is included in
            the returned sequence.
        temperature (float): Softmax temperature. 0 means greedy (argmax).
        top_p (float): Nucleus threshold; 1.0 disables filtering.

    Returns:
        Token IDs of shape (1, seq_len + generated).
    """
    model_was_training = model.training
    model.eval()

    for _ in range(max_tokens):
        logits = model(prompt[:, -max_context_len:])
        last_logit = logits[:, -1, :]

        if temperature == 0:
            result = torch.argmax(last_logit, dim=-1, keepdim=True)
        else:
            last_logit = softmax(last_logit / temperature, dimension=-1)
            last_logit = top_p_sampling(last_logit, top_p)
            result = torch.multinomial(last_logit, num_samples=1)

        prompt = torch.concat([prompt, result], dim=-1)

        if result.item() == end_token_id:
            break

    if model_was_training:
        model.train()

    return prompt


def build_tokenizer(config: RunConfig) -> Tokenizer:
    """Load the BPE tokenizer that produced this run's training tokens.

    The vocab and merges must be the ones the model was trained on: token id N
    means whatever the pickles say it means, and pairing a checkpoint with the
    wrong tokenizer decodes to fluent-looking nonsense rather than erroring.
    """
    return Tokenizer.from_files(
        vocab_filepath=config.dataset.vocab_path,
        merges_filepath=config.dataset.merges_path,
        special_tokens=SPECIAL_TOKENS,
    )


def load_generator(config: RunConfig) -> tuple[TransformerLM, Tokenizer]:
    """Rebuild the model described by `config` and restore its final checkpoint.

    The architecture fields must match the ones that trained it -- load_state_dict
    is strict about parameters.
    """
    checkpoint = config.checkpoint_path / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} -- train this run first")

    model = build_model(config)

    optimizer = build_optimizer(config, model)
    load_checkpoint(src=checkpoint, model=model, optimizer=optimizer)

    model.eval()
    return model, build_tokenizer(config)


if __name__ == "__main__":
    print("Hello Welcome to your own chatbot. Please select the following models to chat with:")
    for number, (_, description) in enumerate(CHAT_RUNS, start=1):
        print(f"\t{number}. {description}")

    choice = input(f"\nPlease enter your choice here (1-{len(CHAT_RUNS)}): ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(CHAT_RUNS):
        raise SystemExit(f"expected 1-{len(CHAT_RUNS)}, got {choice!r}")

    config = RUNS[CHAT_RUNS[int(choice) - 1][0]]
    model, tokenizer = load_generator(config)
    end_token_id = tokenizer.vocab_to_id[END_OF_TEXT]

    print("\nModel loaded, please begin chatting\n")

    context: list[int] = []

    while True:
        try:
            text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text.strip():
            continue

        context.extend(tokenizer.encode(text))
        prompt = torch.tensor([context], dtype=torch.long, device=config.train.torch_device)

        output = decode(
            model=model,
            prompt=prompt,
            max_context_len=model.context_length,
            max_tokens=MAX_TOKENS,
            end_token_id=end_token_id,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        context = output[0].tolist()
        completion = context[prompt.shape[1]:]

        # decode() returns the token it stopped on, which belongs in the running
        # context but would print a literal "<|endoftext|>" to the user.
        if completion and completion[-1] == end_token_id:
            completion = completion[:-1]

        print(f"bot> {tokenizer.decode(completion).strip()}\n")

        # The model can only attend to context_length tokens; older ones are
        # dead weight, and trimming keeps each turn's forward pass bounded.
        if len(context) > model.context_length:
            context = context[-model.context_length:]
