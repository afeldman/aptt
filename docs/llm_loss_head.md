# LLM Loss Functions und Heads

Implementierung von Language Modeling Loss Functions und Output Heads basierend auf der DeepSeek-V3 Architektur.

## Referenz

DeepSeek-V3 Technical Report: https://arxiv.org/html/2412.19437v2

## Übersicht

Diese Module implementieren die Loss Functions und Output Heads die für Language Modeling Training benötigt werden:

### Loss Functions (`aptt.loss.language_modeling`)

1. **LanguageModelingLoss** - Standard Cross-Entropy Loss für Next-Token Prediction
2. **MultiTokenPredictionLoss** - Multi-Token Prediction (MTP) Loss mit λ-Gewichtung
3. **PerplexityMetric** - Perplexity Metrik für Model Evaluation
4. **TokenAccuracyMetric** - Token-level Accuracy

### Heads (`aptt.heads.language_modeling`)

1. **LanguageModelingHead** - Standard LM Head mit Weight Tying
2. **MultiTokenPredictionHead** - MTP Head für Future Token Prediction
3. **CombinedLMHead** - Kombiniert Main + MTP Heads

## Loss Functions

### LanguageModelingLoss

Standard Cross-Entropy Loss für Language Modeling mit optionalem Label Smoothing.

```python
from aptt.loss.language_modeling import LanguageModelingLoss

# Setup
loss_fn = LanguageModelingLoss(
    vocab_size=50000,
    ignore_index=-100,      # Padding token
    label_smoothing=0.1,    # Optional: verhindert Overconfidence
    reduction='mean'
)

# Forward pass
logits = model(input_ids)  # (batch, seq_len, vocab_size)
labels = target_ids        # (batch, seq_len)
loss = loss_fn(logits, labels)
```

**Parameter:**

- `vocab_size`: Größe des Vokabulars
- `ignore_index`: Token ID die ignoriert werden soll (typisch: padding)
- `label_smoothing`: Smoothing factor (0.0 = kein smoothing)
- `reduction`: 'mean', 'sum', oder 'none'

### MultiTokenPredictionLoss

Kombiniert Main Loss mit zusätzlichen Future Token Predictions für verbesserte Supervision.

```python
from aptt.loss.language_modeling import MultiTokenPredictionLoss

# Setup (DeepSeek-V3 verwendet D=1)
loss_fn = MultiTokenPredictionLoss(
    vocab_size=50000,
    mtp_depth=1,          # Anzahl zusätzlicher Predictions
    mtp_lambda=0.3,       # Gewichtung der MTP Terms
    label_smoothing=0.1
)

# Forward pass
main_logits = model(input_ids)           # (batch, seq_len, vocab_size)
mtp_logits = [model.mtp_head_1(...)]     # List[(batch, seq_len-k, vocab_size)]
labels = target_ids                       # (batch, seq_len)

total_loss, loss_dict = loss_fn(main_logits, mtp_logits, labels)
```

**Loss Berechnung:**

```
total_loss = main_loss + λ * sum(mtp_loss_k for k=1..D)
```

**Returns:**

- `total_loss`: Gewichtete Gesamtloss
- `loss_dict`: Dict mit 'main_loss', 'mtp_loss_k', 'total_loss'

### Metrics

```python
from aptt.loss.language_modeling import PerplexityMetric, TokenAccuracyMetric

# Perplexity (niedrigere Werte = besser)
ppl_metric = PerplexityMetric()
ppl = ppl_metric(logits, labels)  # exp(cross_entropy_loss)

# Token Accuracy
acc_metric = TokenAccuracyMetric()
acc = acc_metric(logits, labels)  # 0.0 bis 1.0
```

## Heads

### LanguageModelingHead

Standard Output Head: Projiziert Hidden States auf Vocabulary Logits.

```python
from aptt.heads.language_modeling import LanguageModelingHead

head = LanguageModelingHead(
    d_model=2048,
    vocab_size=50000,
    bias=False,           # Standard in modernen LLMs
    tie_weights=True      # Weight tying mit Embeddings
)

# Forward
hidden = transformer(input_ids)  # (batch, seq_len, d_model)
logits = head(hidden)            # (batch, seq_len, vocab_size)

# Weight Tying (optional)
embedding = nn.Embedding(vocab_size, d_model)
head.tie_embedding_weights(embedding)
```

**Weight Tying:** Teilt Weights zwischen Input Embeddings und Output Projection. Vorteile:

- Reduziert Parameter Count (~4% weniger für vocab_size=50k, d_model=2048)
- Oft bessere Performance
- Standard in GPT, BERT, LLaMA, DeepSeek

### MultiTokenPredictionHead

MTP Head für Future Token Predictions während Training.

```python
from aptt.heads.language_modeling import MultiTokenPredictionHead

mtp_head = MultiTokenPredictionHead(
    d_model=2048,
    vocab_size=50000,
    mtp_depth=1,    # Anzahl Future Tokens (D)
    bias=False
)

# Forward: List von Hidden States für k=1..D
mtp_hidden = [
    hidden[:, :-1, :],  # k=1: alle außer letzter Position
]
mtp_logits = mtp_head(mtp_hidden)  # List[(batch, seq_len-k, vocab_size)]
```

### CombinedLMHead

Kombiniert Main und MTP Heads für vollständiges Training.

```python
from aptt.heads.language_modeling import CombinedLMHead

head = CombinedLMHead(
    d_model=2048,
    vocab_size=50000,
    use_mtp=True,     # MTP aktivieren
    mtp_depth=1,
    tie_weights=True
)

# Training Mode (mit MTP)
head.train()
main_hidden = transformer(input_ids)
mtp_hidden = [main_hidden[:, :-k, :] for k in range(1, mtp_depth+1)]
main_logits, mtp_logits = head(main_hidden, mtp_hidden)

# Inference Mode (ohne MTP)
head.eval()
main_logits, _ = head(main_hidden, None)
```

## Training Pipeline

Vollständiges Beispiel für einen Training Step:

```python
import torch
from aptt.heads.language_modeling import CombinedLMHead
from aptt.loss.language_modeling import MultiTokenPredictionLoss, PerplexityMetric

# Hyperparameters
vocab_size = 50000
d_model = 2048
mtp_depth = 1
mtp_lambda = 0.3

# Model Components
transformer = YourTransformer(d_model=d_model)
head = CombinedLMHead(
    d_model=d_model,
    vocab_size=vocab_size,
    use_mtp=True,
    mtp_depth=mtp_depth
)
loss_fn = MultiTokenPredictionLoss(
    vocab_size=vocab_size,
    mtp_depth=mtp_depth,
    mtp_lambda=mtp_lambda
)
ppl_metric = PerplexityMetric()

# Training Step
def training_step(batch):
    input_ids = batch['input_ids']      # (batch, seq_len)
    labels = batch['labels']            # (batch, seq_len)

    # 1. Forward through Transformer
    main_hidden = transformer(input_ids)  # (batch, seq_len, d_model)

    # 2. Prepare MTP hidden states
    mtp_hidden = [
        main_hidden[:, :-k, :]
        for k in range(1, mtp_depth + 1)
    ]

    # 3. Head: Hidden -> Logits
    main_logits, mtp_logits = head(main_hidden, mtp_hidden)

    # 4. Compute Loss
    total_loss, loss_dict = loss_fn(main_logits, mtp_logits, labels)

    # 5. Metrics
    ppl = ppl_metric(main_logits, labels)

    # 6. Backward
    total_loss.backward()

    return {
        'loss': total_loss.item(),
        'main_loss': loss_dict['main_loss'].item(),
        'mtp_loss': loss_dict['mtp_loss_1'].item(),
        'perplexity': ppl.item()
    }
```

## Inference

Für Inference wird nur der Main Head benötigt:

```python
# Inference Mode
model.eval()
head.eval()

with torch.no_grad():
    # Nur Main Prediction
    hidden = transformer(input_ids)
    logits, _ = head(hidden, None)  # MTP wird ignoriert

    # Next Token
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
```

## Multi-Token Prediction (MTP)

MTP verbessert Model Quality durch zusätzliche Supervision:

**Konzept:**

- Main Head prediziert Token t+1
- MTP Head(s) predizieren zusätzlich t+2, t+3, ...
- Kombinierter Loss mit λ-Gewichtung

**DeepSeek-V3 Setup:**

- D=1 (prediziert 2 Tokens total: t+1 und t+2)
- λ=0.3 (30% Gewicht auf MTP Loss)

**Vorteile:**

- Bessere Representation Learning
- Stabileres Training
- Höhere Model Quality (niedrigere Perplexity)

**Nachteile:**

- ~33% mehr Compute während Training (für D=1)
- Kein Impact auf Inference Speed (MTP wird nicht verwendet)

## Best Practices

1. **Label Smoothing:** Verwende 0.1 für bessere Generalization
2. **Weight Tying:** Aktiviere für kleinere Models und bessere Performance
3. **MTP Lambda:** Start mit 0.3, tune basierend auf validation metrics
4. **Ignore Index:** Setze auf padding_token_id (-100 ist Standard)
5. **Gradient Clipping:** Verwende mit MTP wegen zusätzlichen Gradients

## Performance

**Memory:**

- Main Head: `vocab_size * d_model * 4` bytes (ohne weight tying)
- MTP Head: `vocab_size * d_model * 4 * mtp_depth` bytes zusätzlich

**Beispiel (DeepSeek-V3):**

- vocab_size=50000, d_model=2048, mtp_depth=1
- Main Head: 400 MB
- MTP Head: 400 MB
- Total: 800 MB für Output Heads

**Training Speed:**

- MTP erhöht Training Time um ~33% (für D=1)
- Keine Auswirkung auf Inference

## Code Quality

Alle Module erfüllen APTT Standards:

- ✅ Ruff: 0 Errors
- ✅ MyPy: Strict Type Checking
- ✅ Google-Style Docstrings
- ✅ Comprehensive Examples
- ✅ Test Coverage (siehe tests/)

## Siehe auch

- `aptt.layers.attention.mla` - Multi-Head Latent Attention
- `examples/llm_loss_head_example.py` - Vollständige Code Beispiele
- `docs/llm_implementation_plan.md` - Gesamte Architektur
