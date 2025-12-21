"""Beispiel zur Verwendung der LLM Loss und Head Komponenten.

Demonstriert die Verwendung von LanguageModelingLoss, MultiTokenPredictionLoss
und den zugehörigen Heads.

Referenz: DeepSeek-V3 Technical Report - https://arxiv.org/html/2412.19437v2
"""

from __future__ import annotations

import torch

from deepsuite.heads.language_modeling import CombinedLMHead, LanguageModelingHead
from deepsuite.loss.language_modeling import (
    LanguageModelingLoss,
    MultiTokenPredictionLoss,
    PerplexityMetric,
    TokenAccuracyMetric,
)


def example_basic_lm_loss() -> None:
    """Einfaches Beispiel für Language Modeling Loss."""
    print("\n=== Basic Language Modeling Loss ===")

    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000

    # Simuliere Model Output
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Loss berechnen
    loss_fn = LanguageModelingLoss(vocab_size=vocab_size, label_smoothing=0.1)
    loss = loss_fn(logits, labels)

    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Loss: {loss.item():.4f}")


def example_mtp_loss() -> None:
    """Beispiel für Multi-Token Prediction Loss."""
    print("\n=== Multi-Token Prediction Loss ===")

    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    mtp_depth = 1

    # Simuliere Model Outputs
    main_logits = torch.randn(batch_size, seq_len, vocab_size)
    mtp_logits = [
        torch.randn(batch_size, seq_len - 1, vocab_size),  # k=1: predict t+1
    ]
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # MTP Loss berechnen
    loss_fn = MultiTokenPredictionLoss(
        vocab_size=vocab_size,
        mtp_depth=mtp_depth,
        mtp_lambda=0.3,
    )
    total_loss, loss_dict = loss_fn(main_logits, mtp_logits, labels)

    print(f"Main logits shape: {main_logits.shape}")
    print(f"MTP logits shapes: {[x.shape for x in mtp_logits]}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Main Loss: {loss_dict['main_loss'].item():.4f}")
    print(f"MTP Loss 1: {loss_dict['mtp_loss_1'].item():.4f}")


def example_combined_head() -> None:
    """Beispiel für Combined LM Head mit MTP."""
    print("\n=== Combined LM Head ===")

    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    d_model = 512
    mtp_depth = 1

    # Simuliere Hidden States
    main_hidden = torch.randn(batch_size, seq_len, d_model)
    mtp_hidden = [
        torch.randn(batch_size, seq_len - 1, d_model),  # k=1
    ]

    # Head erstellen
    head = CombinedLMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        use_mtp=True,
        mtp_depth=mtp_depth,
    )

    # Training mode: mit MTP
    head.train()
    main_logits, mtp_logits = head(main_hidden, mtp_hidden)
    print(f"Training - Main logits: {main_logits.shape}")
    print(f"Training - MTP logits: {[x.shape for x in mtp_logits]}")

    # Inference mode: ohne MTP
    head.eval()
    main_logits_inf, mtp_logits_inf = head(main_hidden, None)
    print(f"Inference - Main logits: {main_logits_inf.shape}")
    print(f"Inference - MTP logits: {mtp_logits_inf}")


def example_metrics() -> None:
    """Beispiel für Perplexity und Accuracy Metriken."""
    print("\n=== Metrics: Perplexity & Accuracy ===")

    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000

    # Simuliere Predictions
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Perplexity
    ppl_metric = PerplexityMetric()
    ppl = ppl_metric(logits, labels)
    print(f"Perplexity: {ppl.item():.2f}")

    # Accuracy
    acc_metric = TokenAccuracyMetric()
    acc = acc_metric(logits, labels)
    print(f"Token Accuracy: {acc.item():.2%}")


def example_full_training_step() -> None:
    """Vollständiges Training Step Beispiel."""
    print("\n=== Full Training Step ===")

    # Hyperparameters
    batch_size = 4
    seq_len = 128
    vocab_size = 50000
    d_model = 2048
    mtp_depth = 1
    mtp_lambda = 0.3

    # 1. Model Output (simuliert)
    main_hidden = torch.randn(batch_size, seq_len, d_model)
    mtp_hidden = [torch.randn(batch_size, seq_len - k, d_model) for k in range(1, mtp_depth + 1)]

    # 2. Head: Hidden States -> Logits
    head = CombinedLMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        use_mtp=True,
        mtp_depth=mtp_depth,
    )
    head.train()
    main_logits, mtp_logits = head(main_hidden, mtp_hidden)

    # 3. Labels
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 4. Loss berechnen
    loss_fn = MultiTokenPredictionLoss(
        vocab_size=vocab_size,
        mtp_depth=mtp_depth,
        mtp_lambda=mtp_lambda,
    )
    total_loss, loss_dict = loss_fn(main_logits, mtp_logits, labels)

    # 5. Metrics
    ppl_metric = PerplexityMetric()
    acc_metric = TokenAccuracyMetric()
    ppl = ppl_metric(main_logits, labels)
    acc = acc_metric(main_logits, labels)

    # 6. Report
    print(f"Batch: {batch_size}, Seq: {seq_len}, Vocab: {vocab_size}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  - Main Loss: {loss_dict['main_loss'].item():.4f}")
    print(f"  - MTP Loss 1: {loss_dict['mtp_loss_1'].item():.4f}")
    print(f"Perplexity: {ppl.item():.2f}")
    print(f"Accuracy: {acc.item():.2%}")

    # 7. Backward (würde in echtem Training gemacht)
    # total_loss.backward()


def example_weight_tying() -> None:
    """Beispiel für Weight Tying zwischen Embedding und Head."""
    print("\n=== Weight Tying ===")

    vocab_size = 10000
    d_model = 512

    # Embedding Layer
    embedding = torch.nn.Embedding(vocab_size, d_model)

    # LM Head
    head = LanguageModelingHead(d_model=d_model, vocab_size=vocab_size, tie_weights=True)

    # Tie weights
    head.tie_embedding_weights(embedding)

    # Verify sie teilen sich die weights
    print(f"Embedding weight shape: {embedding.weight.shape}")
    print(f"Head weight shape: {head.projection.weight.shape}")
    print(f"Weights are shared: {embedding.weight.data_ptr() == head.projection.weight.data_ptr()}")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Loss & Head Beispiele")
    print("=" * 60)

    example_basic_lm_loss()
    example_mtp_loss()
    example_combined_head()
    example_metrics()
    example_full_training_step()
    example_weight_tying()

    print("\n" + "=" * 60)
    print("✓ Alle Beispiele erfolgreich durchgeführt")
    print("=" * 60)
