# APTT – Antons PyTorch Tools

**APTT** (Antons PyTorch Tools) is a modular, extensible deep learning framework designed to streamline training, 
evaluation, and experimentation using [PyTorch Lightning](https://www.pytorchlightning.ai/). It supports a wide range of model architectures, 
loss functions, evaluation metrics, and training utilities—across both vision and audio domains.

## Features

- ✅ Wide range of supported model types (YOLO, ResNet, RNNs, WaveNet, etc.)
- 🧩 Pluggable callbacks (TorchScript export, TensorRT optimization, t-SNE visualization, etc.)
- 🧠 Built-in continual learning and knowledge distillation
- ⚙️ Modular structure (Heads, Losses, Layers, Metrics, Callbacks, etc.)
- 📊 Embedding visualization & analysis tools
- 🗂️ Flexible dataset loaders for audio and image tasks
- 🧪 Unit tests and full documentation with Sphinx

## Project Structure

```bash
aptt/
├── aptt/                  # Core source code (models, callbacks, utils, etc.)
├── tests/                 # Unit tests
├── docs/                  # Sphinx-based documentation
├── README.md              # This file
├── pyproject.toml         # Build system and dependencies
└── LICENSE                # License information
```

---

## Installation
```bash
# Clone the repository
git clone https://github.com/your-user/aptt.git
cd aptt

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# (For Sphinx Documentation)
apt-get install libgraphviz-dev

# Install dependencies
uv install .
```

## Quick Start Example
```python
from aptt.lightning_base.trainer import APTTTrainer

trainer = APTTTrainer(config_path="config.yaml")
trainer.train()
```

## Documentation
To build the documentation locally:

```bash
cd docs
make html
```

The HTML output will be located in docs/_build/html/index.html.

## License
This project is licensed under the MIT License – see the LICENSE file for details.
