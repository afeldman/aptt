"""Example usage of DeepSeek-V3 Mixture-of-Experts (MoE) layers.

This example demonstrates:
1. Basic FFN Expert usage
2. Auxiliary-loss-free Router behavior
3. Standard DeepSeekMoE with shared + routed experts
4. Efficient batched MoE implementation
5. Load balancing analysis
6. Integration in DeepSeekV3 model
"""

import torch

from deepsuite.layers.moe import (
    AuxiliaryLossFreeRouter,
    DeepSeekMoE,
    EfficientDeepSeekMoE,
    FFNExpert,
)
from deepsuite.modules.deepseek import DeepSeekV3


def example_1_ffn_expert() -> None:
    """Example 1: Basic FFN Expert with SwiGLU activation."""
    print("\n=== Example 1: FFN Expert ===")

    d_model = 512
    d_ffn = 2048
    expert = FFNExpert(d_model=d_model, d_ffn=d_ffn)

    # Process some tokens
    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, d_model)

    output = expert(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expert parameters: {sum(p.numel() for p in expert.parameters()):,}")


def example_2_router() -> None:
    """Example 2: Auxiliary-loss-free Router with load balancing."""
    print("\n=== Example 2: Auxiliary-Loss-Free Router ===")

    d_model = 512
    n_routed_experts = 256
    n_expert_per_token = 8
    router = AuxiliaryLossFreeRouter(
        d_model=d_model,
        n_routed_experts=n_routed_experts,
        n_expert_per_token=n_expert_per_token,
    )

    # Route some tokens
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, d_model)

    expert_indices, expert_weights, stats = router(x)
    print(f"Input shape: {x.shape}")
    print(f"Expert indices shape: {expert_indices.shape}")  # [batch, seq_len, K]
    print(f"Expert weights shape: {expert_weights.shape}")  # [batch, seq_len, K]
    print(f"Selected experts for first token: {expert_indices[0, 0]}")
    print(f"Expert weights for first token: {expert_weights[0, 0]}")

    # Load balancing stats
    print("\nLoad Balancing Statistics:")
    print(f"Load balance factor: {stats['load_balance_factor']:.4f}")
    print(f"Min/Max expert load: {stats['min_load']:.0f} / {stats['max_load']:.0f}")

    # Show load distribution
    expert_load = stats["expert_counts"]
    print(f"Load per expert (first 10): {expert_load[:10].tolist()}")


def example_3_deepseek_moe() -> None:
    """Example 3: Standard DeepSeekMoE with shared + routed experts."""
    print("\n=== Example 3: Standard DeepSeekMoE ===")

    d_model = 512
    d_ffn = 2048
    n_shared_experts = 1
    n_routed_experts = 64  # Smaller for demo
    n_expert_per_token = 8

    moe = DeepSeekMoE(
        d_model=d_model,
        d_ffn=d_ffn,
        n_shared_experts=n_shared_experts,
        n_routed_experts=n_routed_experts,
        n_expert_per_token=n_expert_per_token,
    )

    # Process some tokens
    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, d_model)

    output, stats = moe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in moe.parameters()):,}")

    print("\nMoE Statistics:")
    print(f"Load balance factor: {stats['load_balance_factor']:.4f}")
    print(f"Min/Max load: {stats['min_load']:.0f} / {stats['max_load']:.0f}")


def example_4_efficient_moe() -> None:
    """Example 4: Efficient batched MoE implementation."""
    print("\n=== Example 4: Efficient DeepSeekMoE ===")

    d_model = 512
    d_ffn = 2048
    n_shared_experts = 1
    n_routed_experts = 64
    n_expert_per_token = 8

    # Standard MoE
    moe_standard = DeepSeekMoE(
        d_model=d_model,
        d_ffn=d_ffn,
        n_shared_experts=n_shared_experts,
        n_routed_experts=n_routed_experts,
        n_expert_per_token=n_expert_per_token,
    )

    # Efficient MoE
    moe_efficient = EfficientDeepSeekMoE(
        d_model=d_model,
        d_ffn=d_ffn,
        n_shared_experts=n_shared_experts,
        n_routed_experts=n_routed_experts,
        n_expert_per_token=n_expert_per_token,
    )

    # Process same input with both
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, d_model)

    output_standard, stats_standard = moe_standard(x)
    output_efficient, stats_efficient = moe_efficient(x)

    print(f"Input shape: {x.shape}")
    print(f"Standard output shape: {output_standard.shape}")
    print(f"Efficient output shape: {output_efficient.shape}")

    # Check numerical equivalence (should be similar but not identical due to
    # different computation order)
    diff = (output_standard - output_efficient).abs().mean()
    print(f"\nMean absolute difference: {diff:.6f}")
    print("(Small differences expected due to different computation order)")

    print("\nStandard MoE stats:")
    print(f"  Load balance factor: {stats_standard['load_balance_factor']:.4f}")
    print(f"  Min/Max load: {stats_standard['min_load']:.0f} / {stats_standard['max_load']:.0f}")

    print("\nEfficient MoE stats:")
    print(f"  Load balance factor: {stats_efficient['load_balance_factor']:.4f}")
    print(f"  Min/Max load: {stats_efficient['min_load']:.0f} / {stats_efficient['max_load']:.0f}")


def example_5_load_balancing_analysis() -> None:
    """Example 5: Analyze load balancing over multiple steps."""
    print("\n=== Example 5: Load Balancing Analysis ===")

    d_model = 512
    n_routed_experts = 64
    n_expert_per_token = 8
    router = AuxiliaryLossFreeRouter(
        d_model=d_model,
        n_routed_experts=n_routed_experts,
        n_expert_per_token=n_expert_per_token,
    )

    # Process multiple batches
    n_batches = 10
    batch_size, seq_len = 4, 32

    all_loads = []
    for i in range(n_batches):
        x = torch.randn(batch_size, seq_len, d_model)
        _, _, stats = router(x)
        all_loads.append(stats["expert_counts"])

        if i == 0 or i == n_batches - 1:
            print(f"\nBatch {i + 1}:")
            print(f"  Load balance factor: {stats['load_balance_factor']:.4f}")
            print(f"  Min/Max: {stats['min_load']:.0f} / {stats['max_load']:.0f}")

    # Aggregate stats
    all_loads = torch.stack(all_loads)
    mean_load = all_loads.mean(dim=0)
    std_load = all_loads.std(dim=0)

    print(f"\nAggregate statistics over {n_batches} batches:")
    print(f"Mean expert load: {mean_load.mean():.1f} ± {std_load.mean():.1f}")
    print(f"Load std across experts: {mean_load.std():.2f}")


def example_6_deepseek_with_moe() -> None:
    """Example 6: DeepSeek-V3 model with MoE enabled."""
    print("\n=== Example 6: DeepSeek-V3 with MoE ===")

    # Small config for demo
    vocab_size = 5000
    d_model = 512
    n_layers = 4
    n_heads = 8
    max_seq_len = 256

    # Create model with MoE
    model = DeepSeekV3(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        use_moe=True,  # Enable MoE
        n_shared_experts=1,
        n_routed_experts=32,  # Smaller for demo
        n_expert_per_token=4,
    )

    # Forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compare with non-MoE model
    model_no_moe = DeepSeekV3(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        use_moe=False,  # Standard FFN
    )

    print("\nModel comparison:")
    print(f"MoE parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FFN parameters: {sum(p.numel() for p in model_no_moe.parameters()):,}")


def example_7_generation_with_moe() -> None:
    """Example 7: Text generation with MoE-enabled model."""
    print("\n=== Example 7: Generation with MoE ===")

    # Small config
    vocab_size = 1000
    d_model = 256
    n_layers = 2
    n_heads = 4
    max_seq_len = 128

    model = DeepSeekV3(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        use_moe=True,
        n_shared_experts=1,
        n_routed_experts=16,
        n_expert_per_token=4,
    )
    model.eval()

    # Simple greedy generation
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = prompt.clone()

    print(f"Prompt shape: {prompt.shape}")
    print("Generating tokens...")

    with torch.no_grad():
        for i in range(20):
            logits = model(generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if i < 5 or i >= 18:
                print(f"  Step {i + 1}: Generated token {next_token.item()}")
            elif i == 5:
                print("  ...")

    print(f"\nFinal sequence shape: {generated.shape}")
    print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")


def main() -> None:
    """Run all examples."""
    print("=" * 70)
    print("DeepSeek-V3 Mixture-of-Experts (MoE) Examples")
    print("=" * 70)

    example_1_ffn_expert()
    example_2_router()
    example_3_deepseek_moe()
    example_4_efficient_moe()
    example_5_load_balancing_analysis()
    example_6_deepseek_with_moe()
    example_7_generation_with_moe()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
