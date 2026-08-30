from typing import Optional
import torch
from torch import nn
from einops import einsum, reduce
from jaxtyping import Float, Int

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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        token_ids: Int[torch.Tensor, "... seq"]
    ) -> Float[torch.Tensor, "... seq d_model"]:
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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