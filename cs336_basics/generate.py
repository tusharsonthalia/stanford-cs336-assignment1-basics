import torch
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.modules import softmax


def top_p_sampling(
    probs: Float[Tensor, "1 vocab"],
    top_p: float
) -> Float[Tensor, "1 vocab"]:
    """Top-p keeps only the most likely tokens, drops the rest, and rescales what
    remains to sum to 1.

    Args:
        probs: Probability row of shape (1, vocab) -- already normalized, so
            this runs after the softmax, not before. Modified in place.
        top_p (float): Cumulative probability threshold in (0, 1].

    Returns:
        The same tensor, with out-of-nucleus entries zeroed and the survivors
        renormalized to sum to 1.
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    valid_idx = (cumsum <= top_p)
    valid_idx[..., 0] = True

    invalid_idx = sorted_idx[~valid_idx]

    probs[:, invalid_idx] = 0
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
