# LLM Transformer Implementierungsplan für APTT v0.2.0

> **Referenz**: DeepSeek-V3 Technical Report - https://arxiv.org/html/2412.19437v2

## 🎯 Überblick

Basierend auf DeepSeek-V3 (671B Parameter Mixture-of-Experts Sprachmodell) implementieren wir einen modernen Transformer mit folgenden Schlüsselmerkmalen:

- 37B aktivierte Parameter pro Token
- Multi-Head Latent Attention (MLA)
- DeepSeekMoE Architektur
- Auxiliary-Loss-Free Load Balancing
- Multi-Token Prediction (MTP)
- FP8 Mixed Precision Training

---

## 📦 Zu implementierende Komponenten

### 1. **Multi-Head Latent Attention (MLA)**

#### Beschreibung

Effiziente Attention-Mechanismus mit Low-Rank KV-Kompression für reduzierten KV-Cache während der Inferenz.

#### Komponenten

```
aptt/layers/attention/
├── __init__.py
├── mla.py                 # Multi-Head Latent Attention
├── kv_compression.py      # KV Compression mit RoPE
└── rope.py                # Rotary Position Embedding
```

#### Integration in bestehende Struktur

```python
# src/aptt/layers/attention/mla.py
class MultiHeadLatentAttention(nn.Module):
    """
    MLA with low-rank joint compression for keys and values.

    Args:
        d: embedding dimension
        n_h: number of attention heads
        d_h: dimension per head
        d_c: KV compression dimension (<<d_h*n_h)
        d_c_q: query compression dimension
        d_h_R: per-head dimension for RoPE
    """
    def __init__(
        self,
        d: int = 7168,
        n_h: int = 128,
        d_h: int = 128,
        d_c: int = 512,
        d_c_q: int = 1536,
        d_h_R: int = 64
    ):
        super().__init__()
        # Down-projection für KV
        self.W_D_KV = nn.Linear(d, d_c)
        # Up-projection für K und V
        self.W_U_K = nn.Linear(d_c, d_h * n_h)
        self.W_U_V = nn.Linear(d_c, d_h * n_h)
        # RoPE für Keys
        self.W_K_R = nn.Linear(d, d_h_R)
        self.rope = RotaryPositionEmbedding(d_h_R)

        # Query Compression
        self.W_D_Q = nn.Linear(d, d_c_q)
        self.W_U_Q = nn.Linear(d_c_q, d_h * n_h)
        self.W_Q_R = nn.Linear(d_c_q, d_h_R * n_h)

        # Output projection
        self.W_O = nn.Linear(d_h * n_h, d)
```

#### Passt zu existierender Struktur

- ✅ Kann in `src/aptt/layers/` als neues Modul hinzugefügt werden
- ✅ Nutzt existierende `nn.Linear` Bausteine
- ✅ Kompatibel mit `BaseLightningModule` Training

---

### 2. **DeepSeekMoE (Mixture-of-Experts)**

#### Beschreibung

Fine-grained MoE mit:

- N_s shared experts (immer aktiv)
- N_r routed experts (selektiv aktiviert)
- K_r aktivierte routed experts pro Token
- Auxiliary-loss-free load balancing

#### Komponenten

```
aptt/layers/moe/
├── __init__.py
├── deepseek_moe.py        # DeepSeekMoE Layer
├── expert.py              # FFN Expert
├── router.py              # Token-to-Expert Routing
└── load_balancer.py       # Bias-based Load Balancing
```

#### Architektur

```python
# src/aptt/layers/moe/deepseek_moe.py
class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE Layer mit shared + routed experts.

    Args:
        d: hidden dimension
        N_s: number of shared experts
        N_r: number of routed experts
        K_r: number of activated routed experts
        d_ffn: FFN intermediate dimension (2048 in DeepSeek-V3)
    """
    def __init__(
        self,
        d: int = 7168,
        N_s: int = 1,
        N_r: int = 256,
        K_r: int = 8,
        d_ffn: int = 2048
    ):
        super().__init__()
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            FFNExpert(d, d_ffn) for _ in range(N_s)
        ])

        # Routed experts (selectively activated)
        self.routed_experts = nn.ModuleList([
            FFNExpert(d, d_ffn) for _ in range(N_r)
        ])

        # Router mit bias-based load balancing
        self.router = Router(d, N_r, K_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h'_t = u_t + sum(FFN_i^(s)(u_t)) + sum(g_i,t * FFN_i^(r)(u_t))
        output = x

        # Shared experts (immer)
        for expert in self.shared_experts:
            output = output + expert(x)

        # Routed experts (selektiv)
        routing_weights, expert_indices = self.router(x)
        for idx, weight in zip(expert_indices, routing_weights):
            output = output + weight * self.routed_experts[idx](x)

        return output
```

#### Integration in bestehende Struktur

- ✅ Neue Layer-Kategorie neben `aptt/layers/bottleneck.py`, `hermite.py`, etc.
- ✅ Kann in Transformer-Blocks integriert werden (ersetzt Standard-FFN)
- ✅ Kompatibel mit existierendem `BaseModule` Training-Loop

---

### 3. **Auxiliary-Loss-Free Load Balancing**

#### Beschreibung

Bias-basierte Strategie statt Auxiliary Loss:

- Jeder Expert hat bias term `b_i`
- Dynamische Anpassung bei Unbalance: `b_i += γ` (underloaded) oder `b_i -= γ` (overloaded)
- Bias nur für Routing, nicht für Gating Values

#### Komponenten

```python
# src/aptt/layers/moe/load_balancer.py
class AuxiliaryLossFreeBalancer(nn.Module):
    """
    Bias-based load balancing ohne auxiliary loss.

    Args:
        N_r: number of routed experts
        gamma: bias update speed (0.001 in DeepSeek-V3)
    """
    def __init__(self, N_r: int = 256, gamma: float = 0.001):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(N_r))
        self.gamma = gamma

    def update_bias(self, expert_load: torch.Tensor, target_load: float):
        """Update bias based on expert load."""
        overloaded = expert_load > target_load
        underloaded = expert_load < target_load

        with torch.no_grad():
            self.bias[overloaded] -= self.gamma
            self.bias[underloaded] += self.gamma
```

#### Integration

- ✅ Als Teil des `Router` in `DeepSeekMoE`
- ✅ Callback für Bias-Updates nach jedem Training-Step
- ✅ Monitoring mit `BaseLightningModule.log()`

---

### 4. **Multi-Token Prediction (MTP)**

#### Beschreibung

Prediziert D zusätzliche Tokens pro Position:

- D = 1 in DeepSeek-V3 (next 2 tokens total)
- Sequentielle Prediction mit vollständiger Causal Chain
- Shared Embedding & Output Head mit Main Model

#### Komponenten

```
aptt/model/transformer/
├── __init__.py
├── mtp_module.py          # Multi-Token Prediction Module
└── transformer_block.py   # DeepSeek-V3 Transformer Block
```

#### Architektur

```python
# src/aptt/model/transformer/mtp_module.py
class MultiTokenPredictionModule(nn.Module):
    """
    MTP Module für Depth k.

    h'_i^k = M_k * [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]
    h_{1:T-k}^k = TRM_k(h'_{1:T-k}^k)
    P_{i+k+1}^k = OutHead(h_i^k)
    """
    def __init__(
        self,
        d: int = 7168,
        depth: int = 1,
        n_layers: int = 61
    ):
        super().__init__()
        self.depth = depth
        self.projection = nn.Linear(2 * d, d)  # [h; emb] -> h
        self.transformer_block = TransformerBlock(d, n_layers)
        self.rms_norm = nn.RMSNorm(d)

    def forward(self, h_prev: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
        # Combine previous representation + token embedding
        h_combined = self.projection(
            torch.cat([self.rms_norm(h_prev), self.rms_norm(token_emb)], dim=-1)
        )
        # Transform
        h_k = self.transformer_block(h_combined)
        return h_k
```

#### Integration

- ✅ Als optionales Modul in `aptt/model/transformer/`
- ✅ Training mit Multi-Loss (Main + MTP losses)
- ✅ Inference: Discard MTP oder nutze für Speculative Decoding

---

### 5. **FP8 Mixed Precision Training**

#### Beschreibung

- Fine-grained Quantization: 1×128 tiles (activations), 128×128 blocks (weights)
- Increased Accumulation Precision: FP32 accumulation mit CUDA Cores
- E4M3 format für alle Tensoren (Mantissa over Exponents)

#### Komponenten

```
aptt/utils/precision/
├── __init__.py
├── fp8_quantizer.py       # Fine-grained FP8 Quantization
└── fp8_gemm.py            # Custom FP8 GEMM mit FP32 Accumulation
```

#### Hinweis

**FP8 Training ist Hardware-spezifisch (NVIDIA H800/Blackwell)**. Für APPT v0.2.0:

- ✅ API-Design vorbereiten
- ❌ Vorerst keine Full-Implementation (erfordert Custom CUDA Kernels)
- ✅ Platzhalter für zukünftige Integration

---

## 🏗️ Modelstruktur für DeepSeek-V3

### Vollständige Architektur

```
aptt/model/transformer/
├── __init__.py
├── llm.py                 # Main Model
├── transformer_block.py   # Single Transformer Layer
├── mtp_module.py          # Multi-Token Prediction
└── embedding.py           # Token + Position Embeddings

aptt/layers/
├── attention/
│   ├── mla.py
│   ├── kv_compression.py
│   └── rope.py
└── moe/
    ├── deepseek_moe.py
    ├── expert.py
    ├── router.py
    └── load_balancer.py
```

### Beispiel: Vollständiger Transformer Block

```python
# src/aptt/model/transformer/transformer_block.py
class TransformerBlock(nn.Module):
    """
    Single Transformer layer basierend auf DeepSeek-V3 Architektur.

    Attention -> DeepSeekMoE (statt Standard-FFN)
    """
    def __init__(
        self,
        d: int = 7168,
        n_h: int = 128,
        d_h: int = 128,
        N_s: int = 1,
        N_r: int = 256,
        K_r: int = 8
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(d)
        self.attention = MultiHeadLatentAttention(d, n_h, d_h)

        self.ln2 = nn.RMSNorm(d)
        self.moe = DeepSeekMoE(d, N_s, N_r, K_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture
        x = x + self.attention(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x
```

---

## 🔗 Integration in APTT-Datenstruktur

### 1. **Passt zu existierenden Modulen**

| APTT Komponente                    | DeepSeek-V3 Mapping                   |
| ---------------------------------- | ------------------------------------- |
| `aptt/layers/bottleneck.py`        | → Bleibt bestehen (für CNNs)          |
| `aptt/layers/hermite.py`           | → Bleibt bestehen (Signal Processing) |
| **NEU: `aptt/layers/attention/`**  | → MLA, RoPE, KV-Compression           |
| **NEU: `aptt/layers/moe/`**        | → DeepSeekMoE, Router, Load Balancer  |
| `aptt/model/detection/`            | → Bleibt bestehen (CV models)         |
| **NEU: `aptt/model/transformer/`** | → LLM Transformer                     |
| `aptt/lightning_base/module.py`    | → Erweitern für MoE-Training          |

### 2. **Training mit BaseLightningModule**

```python
# src/aptt/modul/llm.py
from aptt.lightning_base.module import BaseLightningModule
from aptt.model.transformer.llm import LLMModel

class LLMModule(BaseLightningModule):
    """
    PyTorch Lightning Module für LLM Transformer (basierend auf DeepSeek-V3).
    """
    def __init__(
        self,
        vocab_size: int = 128000,
        d_model: int = 7168,
        n_layers: int = 61,
        n_heads: int = 128,
        N_r: int = 256,
        K_r: int = 8,
        learning_rate: float = 2.2e-4,
        use_mtp: bool = True,
        mtp_depth: int = 1
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LLMModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            N_r=N_r,
            K_r=K_r,
            use_mtp=use_mtp,
            mtp_depth=mtp_depth
        )

        # Standard Cross-Entropy Loss
        self.criterion = nn.CrossEntropyLoss()

        # MTP Loss (falls aktiviert)
        if use_mtp:
            self.mtp_criterion = nn.CrossEntropyLoss()
            self.mtp_lambda = 0.3  # Initial weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch

        # Main prediction
        logits = self(input_ids)
        main_loss = self.criterion(logits.view(-1, self.hparams.vocab_size), labels.view(-1))

        # MTP loss (if enabled)
        total_loss = main_loss
        if self.hparams.use_mtp:
            mtp_logits = self.model.get_mtp_predictions(input_ids)
            mtp_loss = self.mtp_criterion(
                mtp_logits.view(-1, self.hparams.vocab_size),
                labels[:, 1:].contiguous().view(-1)
            )
            total_loss = main_loss + self.mtp_lambda * mtp_loss

            self.log('train/mtp_loss', mtp_loss, prog_bar=True)

        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/main_loss', main_loss)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # Cosine decay with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
            eta_min=2.2e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
```

### 3. **Callbacks für MoE Load Balancing**

```python
# src/aptt/callbacks/moe_balancer.py
from pytorch_lightning.callbacks import Callback

class MoELoadBalancerCallback(Callback):
    """
    Updates expert bias für auxiliary-loss-free load balancing.
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Get expert load statistics
        for name, module in pl_module.named_modules():
            if isinstance(module, DeepSeekMoE):
                expert_load = module.router.get_expert_load()
                target_load = 1.0 / module.N_r  # Balanced load

                # Update bias
                module.router.load_balancer.update_bias(expert_load, target_load)

                # Log load imbalance
                load_imbalance = expert_load.std().item()
                trainer.logger.log_metrics({
                    f'{name}/load_imbalance': load_imbalance
                }, step=trainer.global_step)
```

---

## 📊 Metriken & Evaluation

### Neue Metriken für Transformer-Training

```python
# src/aptt/metric/language_modeling.py
import torch
import torch.nn as nn

class PerplexityMetric(nn.Module):
    """Perplexity = exp(CrossEntropyLoss)."""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        return torch.exp(loss)

class TokenAccuracy(nn.Module):
    """Token-level accuracy."""
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        return correct.mean()
```

---

## 🚀 Implementierungsreihenfolge für v0.2.0

### Phase 1: Core Attention (Woche 1-2)

1. ✅ `aptt/layers/attention/rope.py` - Rotary Position Embedding
2. ✅ `aptt/layers/attention/kv_compression.py` - KV Compression
3. ✅ `aptt/layers/attention/mla.py` - Multi-Head Latent Attention
4. ✅ Tests + Benchmarks

### Phase 2: MoE Infrastructure (Woche 3-4)

1. ✅ `aptt/layers/moe/expert.py` - FFN Expert
2. ✅ `aptt/layers/moe/router.py` - Token-to-Expert Routing
3. ✅ `aptt/layers/moe/load_balancer.py` - Bias-based Balancing
4. ✅ `aptt/layers/moe/deepseek_moe.py` - DeepSeekMoE Layer
5. ✅ Tests + Load Balancing Validierung

### Phase 3: Transformer Architecture (Woche 5-6)

1. ✅ `aptt/model/transformer/transformer_block.py` - Single Layer
2. ✅ `aptt/model/transformer/embedding.py` - Token + Position Embeddings
3. ✅ `aptt/model/transformer/deepseek_v3.py` - Complete Model
4. ✅ Integration Tests

### Phase 4: Multi-Token Prediction (Woche 7)

1. ✅ `aptt/model/transformer/mtp_module.py` - MTP Module
2. ✅ Integration in DeepSeekV3Model
3. ✅ MTP Training Loop + Loss

### Phase 5: Lightning Module + Training (Woche 8)

1. ✅ `aptt/modules/deepseek_v3.py` - Lightning Module
2. ✅ `aptt/callbacks/moe_balancer.py` - Load Balancing Callback
3. ✅ Training Scripts + Examples
4. ✅ Documentation

### Phase 6: FP8 Preparation (Optional, Woche 9)

1. ✅ `aptt/utils/precision/` - API Design
2. ⚠️ Placeholder Implementation (Custom CUDA benötigt H800)

---

## 🔍 Technische Details

### Memory Footprint

- **MLA KV-Cache Reduction**: `d_c = 512` vs `d_h * n_h = 16384` → **32x kleiner**
- **MoE Efficiency**: Nur `K_r=8` von `N_r=256` experts aktiv → **32x weniger Compute**
- **Total Parameters**: 671B (37B aktiviert) → Trainierbar auf Multi-GPU

### Training Requirements (für kleines Modell)

- **GPU**: 8x A100 80GB oder besser
- **RAM**: 512GB+ System RAM
- **Storage**: 500GB+ für Checkpoints
- **Batch Size**: Gradient accumulation für effektive Batch Size 15360

### Skalierung von DeepSeek-V3 Parametern

Für kleinere Experimente (Proof-of-Concept):

```python
# Small-Scale Config (Demo-Zwecke)
small_config = {
    'd_model': 512,         # statt 7168
    'n_layers': 12,         # statt 61
    'n_heads': 8,           # statt 128
    'N_r': 16,              # statt 256
    'K_r': 2,               # statt 8
}
# → ca. 150M parameters (trainierbar auf 1x A100)
```

---

## 📚 Literatur & Referenzen

1. **DeepSeek-V3 Technical Report**: https://arxiv.org/html/2412.19437v2
2. **RoPE (Rotary Position Embedding)**: Su et al., 2024
3. **MoE (Mixture-of-Experts)**: Lepikhin et al., 2021 (GShard)
4. **FP8 Training**: Micikevicius et al., 2022

---

## ✅ Checkliste für v0.2.0 Release

- [ ] Core Attention Layers implementiert
- [ ] MoE Infrastructure vollständig
- [ ] Transformer Architecture komplett
- [ ] MTP Module funktionsfähig
- [ ] Lightning Module + Training Loop
- [ ] Unit Tests (>80% Coverage)
- [ ] Dokumentation (API + Examples)
- [ ] Beispiel-Notebook für Toy Model
- [ ] Performance Benchmarks
- [ ] pyproject.toml Update (Version 0.2.0)

---

## 🎯 Ziel für v0.2.0

**Moderner LLM Transformer Framework** (inspiriert von DeepSeek-V3) mit:

- ✅ Effiziente MLA Attention
- ✅ Flexible MoE Layer
- ✅ Auxiliary-Loss-Free Balancing
- ✅ Multi-Token Prediction
- ✅ Skalierbar von 100M → 100B+ Parameters
- ✅ Kompatibel mit existierendem APTT-Ökosystem

**Proof-of-Concept**: Trainiere 150M-Parameter Modell auf kleinem Corpus (Wikipedia, C4) als Demonstration der Architektur.

---

## 🛠️ Nächste Schritte

1. **Erstelle Issue Tracker** für jede Phase
2. **Branch erstellen**: `feature/llm-transformer`
3. **Start mit Phase 1** (RoPE + MLA)
4. **Iteratives Development** mit Tests nach jeder Phase
5. **Documentation as you go** (wichtig für v0.2.0 Release!)

---

**Status**: 📝 Planning Phase  
**Target Release**: APTT v0.2.0  
**ETA**: Q2 2025
