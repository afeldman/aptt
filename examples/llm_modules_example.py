"""Beispiel für DeepSeek-V3 und GPT Module.

Demonstriert die Verwendung der LLM Module mit PyTorch Lightning.
"""

from __future__ import annotations

import torch

from deepsuite.modules.deepseek import DeepSeekModule, DeepSeekV3
from deepsuite.modules.gpt import GPT, GPTModule


def example_deepseek_v3_basic() -> None:
    """Grundlegendes DeepSeek-V3 Beispiel."""
    print("\n=== DeepSeek-V3 Basic Example ===")

    # Model erstellen (kleine Version für Demo)
    model = DeepSeekV3(
        vocab_size=10000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        d_head=64,
        max_seq_len=128,
        use_mtp=True,
        mtp_depth=1,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    input_ids = torch.randint(0, 10000, (2, 64))
    model.eval()

    with torch.no_grad():
        # Inference mode (kein MTP)
        logits = model(input_ids, return_mtp=False)
        print(f"Output logits shape: {logits.shape}")

        # Training mode (mit MTP)
        model.train()
        main_logits, mtp_logits = model(input_ids, return_mtp=True)
        print(f"Main logits shape: {main_logits.shape}")
        print(f"MTP logits shapes: {[x.shape for x in mtp_logits]}")


def example_deepseek_generation() -> None:
    """Text Generation mit DeepSeek-V3."""
    print("\n=== DeepSeek-V3 Text Generation ===")

    # Model
    model = DeepSeekV3(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=128,
    )
    model.eval()

    # Generate
    prompt = torch.randint(0, 1000, (1, 10))
    print(f"Prompt length: {prompt.shape[1]}")

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )

    print(f"Generated length: {generated.shape[1]}")
    print(f"New tokens: {generated.shape[1] - prompt.shape[1]}")


def example_deepseek_lightning_module() -> None:
    """DeepSeek Lightning Module."""
    print("\n=== DeepSeek Lightning Module ===")

    # Lightning Module
    module = DeepSeekModule(
        vocab_size=10000,
        d_model=512,
        n_layers=6,
        n_heads=8,
        learning_rate=3e-4,
        use_mtp=True,
        mtp_depth=1,
        mtp_lambda=0.3,
    )

    print("Model: DeepSeek-V3")
    print(f"Parameters: {sum(p.numel() for p in module.parameters()):,}")
    print(f"Use MTP: {module.hparams.use_mtp}")
    print(f"MTP Depth: {module.hparams.mtp_depth}")

    # Simuliere Training Step
    batch = {
        "input_ids": torch.randint(0, 10000, (4, 128)),
        "labels": torch.randint(0, 10000, (4, 128)),
    }

    module.train()
    loss = module.training_step(batch, batch_idx=0)
    print(f"Training loss: {loss.item():.4f}")


def example_gpt_basic() -> None:
    """Grundlegendes GPT Beispiel."""
    print("\n=== GPT Basic Example ===")

    # GPT-2 Small Architektur
    model = GPT(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ffn=3072,
        max_seq_len=1024,
    )

    print("Model: GPT-2 Small")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward
    input_ids = torch.randint(0, 50257, (2, 128))
    model.eval()

    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output shape: {logits.shape}")


def example_gpt_generation() -> None:
    """Text Generation mit GPT."""
    print("\n=== GPT Text Generation ===")

    # Kleineres Modell für Demo
    model = GPT(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=256,
    )
    model.eval()

    # Generate
    prompt = torch.randint(0, 1000, (1, 15))
    print(f"Prompt length: {prompt.shape[1]}")

    generated = model.generate(
        prompt,
        max_new_tokens=30,
        temperature=0.9,
        top_k=40,
    )

    print(f"Generated length: {generated.shape[1]}")
    print(f"New tokens: {generated.shape[1] - prompt.shape[1]}")


def example_gpt_lightning_module() -> None:
    """GPT Lightning Module."""
    print("\n=== GPT Lightning Module ===")

    # Lightning Module
    module = GPTModule(
        vocab_size=10000,
        d_model=512,
        n_layers=6,
        n_heads=8,
        learning_rate=3e-4,
        max_seq_len=512,
    )

    print("Model: GPT")
    print(f"Parameters: {sum(p.numel() for p in module.parameters()):,}")

    # Simuliere Training Step
    batch = {
        "input_ids": torch.randint(0, 10000, (4, 128)),
        "labels": torch.randint(0, 10000, (4, 128)),
    }

    module.train()
    loss = module.training_step(batch, batch_idx=0)
    print(f"Training loss: {loss.item():.4f}")


def example_model_comparison() -> None:
    """Vergleich DeepSeek-V3 vs GPT."""
    print("\n=== Model Comparison ===")

    config = {
        "vocab_size": 10000,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "max_seq_len": 256,
    }

    # DeepSeek-V3
    deepseek = DeepSeekV3(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        use_mtp=False,  # Disable for fair comparison
    )

    # GPT
    gpt = GPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
    )

    deepseek_params = sum(p.numel() for p in deepseek.parameters())
    gpt_params = sum(p.numel() for p in gpt.parameters())

    print(f"Configuration: {config}")
    print(f"\nDeepSeek-V3 Parameters: {deepseek_params:,}")
    print(f"GPT Parameters: {gpt_params:,}")
    print(f"Difference: {abs(deepseek_params - gpt_params):,}")
    print("\nKey Differences:")
    print("  - DeepSeek: Multi-Head Latent Attention (MLA) with KV-Compression")
    print("  - DeepSeek: Optional Multi-Token Prediction (MTP)")
    print("  - DeepSeek: Ready for MoE integration (Phase 2)")
    print("  - GPT: Standard Multi-Head Attention")
    print("  - GPT: Absolute Position Embeddings")


def example_parameter_efficiency() -> None:
    """Demonstriere Parameter Efficiency von MLA."""
    print("\n=== Parameter Efficiency: MLA vs Standard Attention ===")

    # Große Model Configuration
    d_model = 2048
    n_heads = 32
    vocab_size = 50000

    # DeepSeek with MLA
    deepseek = DeepSeekV3(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=6,
        n_heads=n_heads,
        d_kv_compression=512,  # Low-rank compression
        use_mtp=False,
    )

    # GPT with standard attention
    gpt = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=6,
        n_heads=n_heads,
    )

    ds_params = sum(p.numel() for p in deepseek.parameters())
    gpt_params = sum(p.numel() for p in gpt.parameters())

    print(f"d_model: {d_model}, n_heads: {n_heads}, vocab_size: {vocab_size}")
    print(f"\nDeepSeek-V3 (MLA): {ds_params:,} parameters")
    print(f"GPT (Standard): {gpt_params:,} parameters")
    print(f"Reduction: {(1 - ds_params / gpt_params) * 100:.1f}%")
    print("\nNote: MLA reduces KV-cache size by ~32x during inference!")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Module Beispiele: DeepSeek-V3 & GPT")
    print("=" * 60)

    example_deepseek_v3_basic()
    example_deepseek_generation()
    example_deepseek_lightning_module()

    example_gpt_basic()
    example_gpt_generation()
    example_gpt_lightning_module()

    example_model_comparison()
    example_parameter_efficiency()

    print("\n" + "=" * 60)
    print("✓ Alle Beispiele erfolgreich durchgeführt")
    print("=" * 60)
