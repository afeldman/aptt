"""Example script demonstrating APTT coding standards.

This module shows how to write code that passes all Ruff and MyPy checks
with proper Google-style docstrings and type annotations.

Reference:
    APTT Coding Standards - docs/coding_standards.md
"""

from __future__ import annotations

import torch
from torch import nn

from deepsuite.layers.attention import MultiHeadLatentAttention


class ExampleTransformerLayer(nn.Module):
    """Example Transformer layer demonstrating coding standards.

    This layer combines Multi-Head Latent Attention with a feed-forward network
    to create a complete Transformer block.

    Args:
        d_model: Model dimension (embedding size).
        n_heads: Number of attention heads.
        d_ff: Feed-forward network dimension.
        dropout: Dropout probability. Defaults to 0.1.
        activation: Activation function for FFN. Defaults to "gelu".

    Attributes:
        attention: Multi-Head Latent Attention layer.
        ffn: Two-layer feed-forward network.
        norm1: Layer normalization before attention.
        norm2: Layer normalization before FFN.
        dropout1: Dropout after attention.
        dropout2: Dropout after FFN.

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)

    Examples:
        >>> # Create transformer layer
        >>> layer = ExampleTransformerLayer(d_model=512, n_heads=8, d_ff=2048)
        >>>
        >>> # Forward pass
        >>> x = torch.randn(2, 100, 512)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([2, 100, 512])
        >>>
        >>> # With attention mask
        >>> mask = torch.ones(2, 100, 100)
        >>> output = layer(x, attention_mask=mask)

    Note:
        This layer uses Pre-LN architecture (LayerNorm before sublayers) which
        is more stable for deep models compared to Post-LN.

    References:
        Vaswani et al. (2017). Attention is All You Need.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Initialize transformer layer."""
        super().__init__()

        # Attention
        self.attention = MultiHeadLatentAttention(
            d=d_model,
            n_h=n_heads,
            d_c=512,
            d_c_q=1536,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Examples:
            >>> layer = ExampleTransformerLayer(d_model=512, n_heads=8, d_ff=2048)
            >>> x = torch.randn(2, 100, 512)
            >>> output = layer(x)
        """
        # Attention sublayer (Pre-LN)
        attn_output, _ = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout1(attn_output)

        # FFN sublayer (Pre-LN)
        ffn_output = self.ffn(self.norm2(x))
        return x + self.dropout2(ffn_output)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation: Name of activation function ("relu", "gelu", "silu").

        Returns:
            Activation module.

        Raises:
            ValueError: If activation name is not supported.

        Examples:
            >>> layer = ExampleTransformerLayer(512, 8, 2048)
            >>> act = layer._get_activation("gelu")
            >>> print(type(act))
            <class 'torch.nn.modules.activation.GELU'>
        """
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()

        msg = f"Unsupported activation: {activation}"
        raise ValueError(msg)

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters.

        Returns:
            Total parameter count.

        Examples:
            >>> layer = ExampleTransformerLayer(d_model=512, n_heads=8, d_ff=2048)
            >>> print(layer.num_parameters)
            1052672
        """
        return sum(p.numel() for p in self.parameters())


def create_position_encoding(
    seq_len: int,
    d_model: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create sinusoidal position encoding.

    Args:
        seq_len: Sequence length.
        d_model: Model dimension.
        device: Device to create encoding on. Defaults to "cpu".

    Returns:
        Position encoding of shape (seq_len, d_model).

    Examples:
        >>> # Create position encoding
        >>> pe = create_position_encoding(100, 512)
        >>> print(pe.shape)
        torch.Size([100, 512])
        >>>
        >>> # Use in model
        >>> x = torch.randn(2, 100, 512)
        >>> x = x + pe.unsqueeze(0)

    Note:
        This is the classical sinusoidal position encoding from the
        original Transformer paper. For relative position encoding,
        consider using RoPE instead.
    """
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * -(torch.log(torch.tensor(10000.0)) / d_model)
    )

    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def apply_gradient_clipping(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Apply gradient clipping to model parameters.

    Args:
        model: PyTorch model.
        max_norm: Maximum gradient norm. Defaults to 1.0.
        norm_type: Type of norm to use (2 for L2 norm). Defaults to 2.0.

    Returns:
        Total norm of gradients before clipping.

    Examples:
        >>> model = nn.Linear(10, 5)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss = model(torch.randn(2, 10)).sum()
        >>> loss.backward()
        >>> total_norm = apply_gradient_clipping(model, max_norm=1.0)
        >>> optimizer.step()

    Note:
        Gradient clipping is essential for training large language models
        to prevent gradient explosion.
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type,
    )


def count_model_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
) -> int:
    """Estimate FLOPs for a forward pass.

    Args:
        model: PyTorch model.
        input_shape: Shape of input tensor (including batch dimension).

    Returns:
        Estimated FLOPs (floating point operations).

    Examples:
        >>> model = ExampleTransformerLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> flops = count_model_flops(model, (1, 100, 512))
        >>> print(f"FLOPs: {flops:,}")

    Warning:
        This is an estimate and may not be exact for all operations.
    """
    # This is a simplified implementation
    # In practice, use libraries like fvcore or thop for accurate counting
    total_flops = 0

    def count_linear_flops(module: nn.Linear, input_size: tuple[int, ...]) -> int:
        """Count FLOPs for linear layer."""
        batch_size = input_size[0]
        in_features = module.in_features
        out_features = module.out_features
        return batch_size * in_features * out_features * 2  # multiply-add

    for module in model.modules():
        if isinstance(module, nn.Linear):
            total_flops += count_linear_flops(module, input_shape)

    return total_flops


if __name__ == "__main__":
    # Example usage demonstrating all features
    print("=== APTT Coding Standards Example ===\n")

    # Create model
    model = ExampleTransformerLayer(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
    print(f"Model created with {model.num_parameters:,} parameters\n")

    # Forward pass
    batch_size, seq_len, d_model = 2, 100, 512
    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")

    # Position encoding
    pe = create_position_encoding(seq_len, d_model)
    print(f"Position encoding shape: {pe.shape}\n")

    # FLOPs estimation
    flops = count_model_flops(model, (batch_size, seq_len, d_model))
    print(f"Estimated FLOPs: {flops:,}\n")

    # Test gradient clipping
    loss = output.sum()
    loss.backward()
    total_norm = apply_gradient_clipping(model, max_norm=1.0)
    print(f"Gradient norm before clipping: {total_norm:.4f}\n")

    print("All checks passed! ✓")
