import torch
from torch import nn
from einops import einsum, reduce
from jaxtyping import Float, Int, Bool

class Linear(nn.Module):
    """A bias-free linear transformation, y = x @ W.T.

    Follows the interface of torch.nn.Linear, without the bias term.

    Args:
        in_features (int): Size of the last dimension of the input.
        out_features (int): Size of the last dimension of the output.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Shape (out_features, in_features) -- i.e. W, not
            W-transpose. Each row holds the weights of a single output unit, so a
            row is contiguous in row-major memory.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # (d_out, d_in): output units first, so W[j] is one contiguous row
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # truncated normal, sigma^2 = 2 / (d_in + d_out), cut at +/- 3 sigma
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        """Apply the transformation. Accepts any number of leading batch dims."""
        # contract over d_in; d_out survives -> equivalent to x @ W.T
        return einsum(x, self.weight,"... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    """
    Token embedding table; maps integer token IDs to dense vectors.

    Follows the interface of torch.nn.Embedding. The forward pass is a row
    gather.

    Args:
        num_embeddings (int): Vocabulary size.
        embedding_dim (int): Dimension of each embedding vector (d_model).
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Shape (num_embeddings, embedding_dim). Vocab first
            so that each embedding vector is a contiguous row.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # (vocab, d_model): rows are gathered, so d_model must be the last axis
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # unit-variance truncated normal, cut at +/- 3
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(
        self,
        token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... d_model"]:
        """Look up embeddings for a batch of token IDs.

        Args:
            token_ids: Integer tensor of any shape, values in [0, num_embeddings).

        Returns:
            Tensor of shape token_ids.shape + (embedding_dim,).
        """
        # advanced indexing: gathers along dim 0 and substitutes token_ids' shape
        # for the vocab axis. Backward is a scatter-add, so repeated IDs accumulate.
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """Root Mean Square layer normalization:  RMSNorm(a_i) = a_i / RMS(a) * g_i,
    where RMS(a) = sqrt(mean(a^2) + eps).

    Normalizes over the LAST axis only; every leading dimension is treated as a
    batch dimension. Unlike LayerNorm there is no mean subtraction and no bias.

    Args:
        d_model (int): Size of the normalized (last) dimension.
        eps (float): Added inside the radicand for numerical stability.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Learned per-feature gain of shape (d_model,),
            initialized to ones so the layer starts as a pure normalizer.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """Normalize x over its last dimension, returning the original dtype."""
        # squaring in fp16/bf16 overflows, so normalize in fp32 and cast back
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_sq = reduce(x ** 2,"... d_model -> ... 1", reduction="mean")
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        result = x * inv_rms * self.weight

        return result.to(in_dtype)

def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    """SiLU / Swish activation: x * sigmoid(x).

    Uses torch.sigmoid rather than the algebraically equivalent x / (1 + exp(-x)),
    which overflows to inf for x < -88 in fp32 (x < -11 in fp16) and produces a
    NaN gradient there.
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU position-wise feed-forward network:

        FFN(x) = W2 @ (SiLU(W1 @ x) * W3 @ x)

    A Gated Linear Unit: w1 and w3 are two independent projections of the same
    input, multiplied elementwise, so the layer computes a product of linear
    readouts rather than a pointwise function of one. w3 acts as a data-dependent
    gate on the w1 branch.

    Operates position-wise -- any number of leading batch dims pass through
    untouched. No bias terms.

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Inner dimension. Supplied by the caller. The canonical
            choice is d_ff ~= (8/3) * d_model rounded to a multiple of 64, which
            keeps the three matrices at the same parameter count as the original
            two-matrix FFN's 8 * d_model^2.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        w1 (Linear): (d_ff, d_model) up-projection; its output receives the SiLU.
        w2 (Linear): (d_model, d_ff) down-projection.
        w3 (Linear): (d_ff, d_model) up-projection; the gate. Shape-identical to
            w1.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """Apply the gated feed-forward transform. Shape is preserved."""
        # w1 branch is activated, w3 branch is the gate
        gated = silu(self.w1(x)) * self.w3(x)
        activated = self.w2(gated)

        return activated


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Splits the feature axis into d_k/2 ADJACENT pairs and rotates each pair as a
    2D vector by an angle proportional to the token's position:

        theta(i, k) = i / Theta^(2k / d_k)      k = 0 .. d_k/2 - 1

    Because a rotation's transpose is its inverse and rotations compose additively,
    (R_i q) . (R_j k) = q^T R_(j-i) k -- so attention LOGITS depend only on the
    relative offset j-i, never on absolute positions.

    Pairing is interleaved -- (x0,x1), (x2,x3), ...
    Has no learnable parameters.

    Args:
        theta (float): Base Theta of the frequency ladder.
        d_k (int): PER-HEAD dimension (d_model // num_heads), not d_model. Must be
            even.
        max_seq_len (int): Number of position rows to precompute.
        device (torch.device | None): Device to build the tables on.
        dtype (torch.dtype | None): Stored dtype of the tables.

    Attributes:
        sin_table, cos_table (Tensor): Shape (max_seq_len, d_k/2), registered with
            persistent=False -- they are fully derived from the constructor args.
    """
    sin_table: Float[torch.Tensor, "max_seq_len d_k/2"]
    cos_table: Float[torch.Tensor, "max_seq_len d_k/2"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0, f"d_k should be even. It is currently {d_k}"
        if dtype is None:
            dtype = torch.get_default_dtype()

        # angles reach thousands of radians at long context, where a float32 ulp
        # is ~1e-4; compute in float64 and cast only the bounded sin/cos results
        seq_positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float64)

        # pair_idx = 0, 2, 4, ...
        pair_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float64)
        inv_freq = theta ** (-pair_idx / d_k)

        # outer product -> (max_seq_len, d_k/2)
        angles = seq_positions.reshape(-1, 1) * inv_freq.reshape(1, -1)

        self.register_buffer("sin_table", torch.sin(angles).to(dtype=dtype), persistent=False)
        self.register_buffer("cos_table", torch.cos(angles).to(dtype=dtype), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """Rotate the pairs of x according to each token's position.

        Args:
            x: Shape (..., seq_len, d_k). Any number of leading batch/head dims.
            token_positions: Shape (..., seq_len), integer. The ABSOLUTE position
                of each token.

        Returns:
            Tensor of the same shape and dtype as x.
        """
        # gather one table row per token: (..., seq_len, d_k/2)
        sin = self.sin_table[token_positions]
        cos = self.cos_table[token_positions]

        # (..., seq_len, d_k/2)
        x_even = x[..., ::2]    # first element of each pair
        x_odd  = x[..., 1::2]   # second element of each pair

        # [[cos, -sin], [sin, cos]] rotation matrix applied to the pair
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # interleave back to [e0, o0, e1, o1, ...].
        # Shape: (..., seq, d_k)
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

def softmax(x: Float[torch.Tensor, "..."], dimension: int) -> Float[torch.Tensor, "..."]:
    """Numerically stable softmax along `dimension`.

    Subtracts the max before exponentiating. softmax is invariant to adding a
    constant to all inputs, so this changes nothing mathematically while keeping
    the largest exponent at exp(0) = 1 -- without it, exp overflows to inf for
    inputs above ~88 in fp32 and inf/inf gives NaN.

    Args:
        x: Tensor of any shape.
        dimension (int): Axis to normalize over.

    Returns:
        Tensor of the same shape; `dimension` sums to 1.
    """
    z = x - torch.amax(x, dim=dimension, keepdim=True)

    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=dimension, keepdim=True)

    return exp_z / sum_exp

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"],
    V: Float[torch.Tensor, "... keys d_v"],
    mask: Bool[torch.Tensor, "queries keys"] | None = None
) -> Float[torch.Tensor, "... queries d_v"]:
    """Scaled dot-product attention:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Maps n queries to n outputs by pooling over m keys.

    Args:
        Q: Shape (..., queries, d_k).
        K: Shape (..., keys, d_k).
        V: Shape (..., keys, d_v).
        mask: Optional boolean mask; False positions are set to -inf so
            they receive zero probability after the softmax.

    Returns:
        Tensor of shape (..., queries, d_v).
    """
    d_k = Q.shape[-1]
    scale = 1 / (d_k ** 0.5)
    qk_proj = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) * scale

    if mask is not None:
        qk_proj = torch.where(mask, qk_proj, -torch.inf)

    probabilities = softmax(qk_proj, dimension=-1)

    attention = einsum(
        probabilities, V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )

    return attention