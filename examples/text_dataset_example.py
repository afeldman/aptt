"""Beispiele für Text/LLM Datasets.

Demonstriert verschiedene Datenformate und Verwendung mit
GPT und DeepSeek-V3 Modellen.
"""

import json
from pathlib import Path
import tempfile

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import torch

from aptt.lightning_base.dataset.text_loader import TextDataLoader, TextDataset


def create_simple_tokenizer(vocab_size: int = 1000) -> Tokenizer:
    """Create a simple BPE tokenizer for demo.

    Args:
        vocab_size: Vocabulary size.

    Returns:
        Trained tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # noqa: S106
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # Sample training data
    texts = [
        "Hello world, this is a test.",
        "Language models are powerful.",
        "DeepSeek and GPT are transformer models.",
        "Machine learning is amazing.",
        "Natural language processing with transformers.",
    ] * 100  # Repeat for better training

    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer


def example_1_text_file() -> None:
    """Example 1: Load from raw text file."""
    print("\n=== Example 1: Raw Text File ===")

    # Create temp text file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            """The transformer architecture has revolutionized natural language processing.
It uses self-attention mechanisms to process sequences in parallel.
GPT models are autoregressive language models based on transformers.
DeepSeek-V3 uses Multi-Head Latent Attention and Mixture-of-Experts.
These models can generate coherent text and solve complex tasks.
"""
        )
        temp_path = f.name

    # Create tokenizer
    tokenizer = create_simple_tokenizer(vocab_size=500)

    # Create dataset
    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=32,
        stride=16,  # Overlap for more samples
    )

    print(f"Created dataset with {len(dataset)} samples")
    print("Sample 0:")

    sample = dataset[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  input_ids: {sample['input_ids'][:10].tolist()}")
    print(f"  labels: {sample['labels'][:10].tolist()}")

    # Decode
    input_text = tokenizer.decode(sample["input_ids"].tolist())
    print(f"  decoded input: {input_text[:100]}...")

    # Cleanup
    Path(temp_path).unlink()


def example_2_jsonl_file() -> None:
    """Example 2: Load from JSONL file."""
    print("\n=== Example 2: JSONL File ===")

    # Create temp JSONL file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        samples = [
            {"text": "First document about transformers.", "metadata": {"id": 1}},
            {"text": "Second document about attention.", "metadata": {"id": 2}},
            {"text": "Third document about language models.", "metadata": {"id": 3}},
        ]
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
        temp_path = f.name

    tokenizer = create_simple_tokenizer(vocab_size=500)

    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=16,
        stride=16,
    )

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Sample 0 input_ids shape: {dataset[0]['input_ids'].shape}")

    # Cleanup
    Path(temp_path).unlink()


def example_3_pretokenized() -> None:
    """Example 3: Load pre-tokenized data."""
    print("\n=== Example 3: Pre-tokenized Data ===")

    # Create pre-tokenized tensor
    token_ids = torch.randint(0, 500, (1000,))  # 1000 random token IDs

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(token_ids, f.name)
        temp_path = f.name

    # Dummy tokenizer (not used for pre-tokenized data)
    tokenizer = create_simple_tokenizer()

    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=64,
        stride=32,
    )

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Sample 0 shape: {dataset[0]['input_ids'].shape}")

    # Cleanup
    Path(temp_path).unlink()


def example_4_mtp_targets() -> None:
    """Example 4: Dataset with Multi-Token Prediction targets."""
    print("\n=== Example 4: Multi-Token Prediction ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("This is a long text for multi-token prediction. " * 20)
        temp_path = f.name

    tokenizer = create_simple_tokenizer()

    # Enable MTP
    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=32,
        stride=32,
        return_mtp=True,
        mtp_depth=3,  # Predict 1, 2, 3 tokens ahead
    )

    print(f"Created MTP dataset with {len(dataset)} samples")

    sample = dataset[0]
    print("Sample 0:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  mtp_labels shape: {sample['mtp_labels'].shape}")

    print("\nMTP targets:")
    for i in range(3):
        print(f"  Lookahead {i+1}: {sample['mtp_labels'][i][:5].tolist()}")

    # Cleanup
    Path(temp_path).unlink()


def example_5_dataloader() -> None:
    """Example 5: Using TextDataLoader with Lightning."""
    print("\n=== Example 5: TextDataLoader ===")

    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.txt"
        val_path = Path(tmpdir) / "val.txt"

        train_path.write_text(
            "Training data for language models. " * 100, encoding="utf-8"
        )
        val_path.write_text(
            "Validation data for evaluation. " * 50, encoding="utf-8"
        )

        tokenizer = create_simple_tokenizer()

        # Create DataLoader
        datamodule = TextDataLoader(
            train_data_path=train_path,
            val_data_path=val_path,
            tokenizer=tokenizer,
            max_seq_len=64,
            batch_size=8,
            num_workers=0,  # 0 for compatibility
            stride=32,
        )

        # Setup
        datamodule.setup("fit")

        print(
            f"Train dataset: {len(datamodule.train_dataset)} samples"
        )
        print(f"Val dataset: {len(datamodule.val_dataset)} samples")

        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Get first batch
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")


def example_6_directory_loading() -> None:
    """Example 6: Load multiple files from directory."""
    print("\n=== Example 6: Directory Loading ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple files
        for i in range(3):
            (Path(tmpdir) / f"file_{i}.txt").write_text(
                f"Document {i}: Some content about language models. " * 20,
                encoding="utf-8",
            )

        tokenizer = create_simple_tokenizer()

        dataset = TextDataset(
            data_path=tmpdir,  # Directory instead of file
            tokenizer=tokenizer,
            max_seq_len=32,
            stride=32,
        )

        print(f"Loaded {len(dataset)} samples from 3 files")


def example_7_with_deepseek() -> None:
    """Example 7: Using dataset with DeepSeek model."""
    print("\n=== Example 7: Training with DeepSeek ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("Training data for DeepSeek model. " * 100)
        temp_path = f.name

    tokenizer = create_simple_tokenizer(vocab_size=1000)

    # Create dataloader with MTP for DeepSeek
    datamodule = TextDataLoader(
        train_data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=4,
        num_workers=0,
        return_mtp=True,  # Enable MTP for DeepSeek
        mtp_depth=1,
    )

    datamodule.setup("fit")

    # Get batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("Batch structure for DeepSeek:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  mtp_labels: {batch['mtp_labels'].shape}")

    print("\nReady for training with DeepSeekModule!")
    print("Example usage:")
    print("  model = DeepSeekModule(..., use_mtp=True)")
    print("  trainer.fit(model, datamodule)")

    # Cleanup
    Path(temp_path).unlink()


def example_8_with_gpt() -> None:
    """Example 8: Using dataset with GPT model."""
    print("\n=== Example 8: Training with GPT ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("Training data for GPT model. " * 100)
        temp_path = f.name

    tokenizer = create_simple_tokenizer(vocab_size=1000)

    # Create dataloader (no MTP for standard GPT)
    datamodule = TextDataLoader(
        train_data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=256,
        batch_size=8,
        num_workers=0,
        return_mtp=False,  # Standard LM for GPT
    )

    datamodule.setup("fit")

    # Get batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("Batch structure for GPT:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    print("\nReady for training with GPTModule!")
    print("Example usage:")
    print("  model = GPTModule(...)")
    print("  trainer.fit(model, datamodule)")

    # Cleanup
    Path(temp_path).unlink()


def main() -> None:
    """Run all examples."""
    print("=" * 70)
    print("Text/LLM Dataset Examples")
    print("=" * 70)

    example_1_text_file()
    example_2_jsonl_file()
    example_3_pretokenized()
    example_4_mtp_targets()
    example_5_dataloader()
    example_6_directory_loading()
    example_7_with_deepseek()
    example_8_with_gpt()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
