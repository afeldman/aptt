"""Einfaches Text Dataset Beispiel ohne externe Abhängigkeiten.

Demonstriert TextDataset mit einem simplen Character-Level Tokenizer.
"""

import json
from pathlib import Path
import tempfile

import torch

from deepsuite.lightning_base.dataset.text_loader import TextDataLoader, TextDataset


class SimpleCharTokenizer:
    """Simple character-level tokenizer for demo.

    Args:
        vocab: List of characters. Defaults to None.
    """

    def __init__(self, vocab: list[str] | None = None) -> None:
        """Initialize tokenizer."""
        if vocab is None:
            # Default: lowercase letters + space + punctuation
            vocab = list("abcdefghijklmnopqrstuvwxyz .,!?")

        self.vocab = vocab
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.id_to_char = dict(enumerate(vocab))
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        text = text.lower()
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text.
        """
        return "".join(self.id_to_char.get(i, "?") for i in token_ids)

    def __call__(self, text: str) -> list[int]:
        """Make tokenizer callable."""
        return self.encode(text)


def example_1_basic_usage() -> None:
    """Example 1: Basic TextDataset usage."""
    print("\n=== Example 1: Basic Usage ===")

    # Create temp text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("hello world. this is a test. language models are great!")
        temp_path = f.name

    # Create tokenizer
    tokenizer = SimpleCharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset
    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=16,
        stride=8,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Get sample
    sample = dataset[0]
    print("\nSample 0:")
    print(f"  input_ids: {sample['input_ids'].tolist()[:10]}")
    print(f"  labels: {sample['labels'].tolist()[:10]}")

    # Decode
    decoded = tokenizer.decode(sample["input_ids"].tolist())
    print(f"  decoded: '{decoded}'")

    # Cleanup
    Path(temp_path).unlink()


def example_2_mtp_dataset() -> None:
    """Example 2: Multi-Token Prediction."""
    print("\n=== Example 2: Multi-Token Prediction ===")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("the quick brown fox jumps over the lazy dog. " * 5)
        temp_path = f.name

    tokenizer = SimpleCharTokenizer()

    # Enable MTP
    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=32,
        stride=32,
        return_mtp=True,
        mtp_depth=3,
    )

    print(f"Dataset size: {len(dataset)} samples")

    sample = dataset[0]
    print("\nSample shapes:")
    print(f"  input_ids: {sample['input_ids'].shape}")
    print(f"  labels: {sample['labels'].shape}")
    print(f"  mtp_labels: {sample['mtp_labels'].shape}")

    print("\nMTP lookahead:")
    for i in range(3):
        decoded = tokenizer.decode(sample["mtp_labels"][i][:10].tolist())
        print(f"  Depth {i + 1}: '{decoded}'")

    # Cleanup
    Path(temp_path).unlink()


def example_3_dataloader() -> None:
    """Example 3: TextDataLoader with batches."""
    print("\n=== Example 3: DataLoader ===")

    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.txt"
        val_path = Path(tmpdir) / "val.txt"

        train_path.write_text(
            "training data for language models. " * 50,
            encoding="utf-8",
        )
        val_path.write_text(
            "validation data for evaluation. " * 30,
            encoding="utf-8",
        )

        tokenizer = SimpleCharTokenizer()

        # Create DataLoader
        datamodule = TextDataLoader(
            train_data_path=train_path,
            val_data_path=val_path,
            tokenizer=tokenizer,
            max_seq_len=64,
            batch_size=4,
            num_workers=0,
            stride=32,
        )

        # Setup
        datamodule.setup("fit")

        print(f"Train samples: {len(datamodule.train_dataset)}")
        print(f"Val samples: {len(datamodule.val_dataset)}")

        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Get first batch
        batch = next(iter(train_loader))
        print("\nBatch structure:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")


def example_4_pretokenized() -> None:
    """Example 4: Pre-tokenized data."""
    print("\n=== Example 4: Pre-tokenized Data ===")

    # Create pre-tokenized tensor
    token_ids = torch.randint(0, 30, (500,))

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(token_ids, f.name)
        temp_path = f.name

    tokenizer = SimpleCharTokenizer()

    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=64,
        stride=32,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample shape: {dataset[0]['input_ids'].shape}")

    # Cleanup
    Path(temp_path).unlink()


def example_5_jsonl() -> None:
    """Example 5: JSONL format."""
    print("\n=== Example 5: JSONL Format ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        samples = [
            {"text": "first document."},
            {"text": "second document."},
            {"text": "third document."},
        ]
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
        temp_path = f.name

    tokenizer = SimpleCharTokenizer()

    dataset = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=16,
        stride=16,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Cleanup
    Path(temp_path).unlink()


def example_6_training_ready() -> None:
    """Example 6: Training-ready dataset for DeepSeek/GPT."""
    print("\n=== Example 6: Training-Ready Dataset ===")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        # Längerer Text für realistisches Training
        text = (
            """
the transformer architecture revolutionized nlp.
it uses self attention mechanisms for sequence processing.
gpt models are autoregressive language models.
deepseek uses multi head latent attention and mixture of experts.
these models can generate coherent text and solve tasks.
"""
            * 20
        )
        f.write(text)
        temp_path = f.name

    tokenizer = SimpleCharTokenizer()

    # For DeepSeek with MTP
    print("\nDataset for DeepSeek (with MTP):")
    dataset_deepseek = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        stride=64,
        return_mtp=True,
        mtp_depth=1,
    )
    print(f"  Samples: {len(dataset_deepseek)}")
    print(f"  Has MTP: {('mtp_labels' in dataset_deepseek[0])}")

    # For GPT (standard LM)
    print("\nDataset for GPT (standard LM):")
    dataset_gpt = TextDataset(
        data_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=256,
        stride=128,
        return_mtp=False,
    )
    print(f"  Samples: {len(dataset_gpt)}")
    print(f"  Has MTP: {('mtp_labels' in dataset_gpt[0])}")

    print("\nReady for training:")
    print("  DeepSeekModule with use_mtp=True")
    print("  GPTModule with standard LM loss")

    # Cleanup
    Path(temp_path).unlink()


def main() -> None:
    """Run all examples."""
    print("=" * 70)
    print("Text Dataset Examples (Simple Tokenizer)")
    print("=" * 70)

    example_1_basic_usage()
    example_2_mtp_dataset()
    example_3_dataloader()
    example_4_pretokenized()
    example_5_jsonl()
    example_6_training_ready()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("\nNote: Diese Beispiele verwenden einen simplen Char-Level Tokenizer.")
    print("Für Production: Verwende tokenizers library (BPE, WordPiece, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
