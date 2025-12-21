# APTT Examples

This directory contains comprehensive examples demonstrating various capabilities of the APTT framework.

## 📁 Available Examples

### 1. Classification Training (`classification_training.py`)

Train image classification models with ResNet backbone on CIFAR-10.

**Features:**

- ResNet-18/34/50 backbone support
- Data augmentation pipeline
- Learning rate scheduling
- Mixed precision training
- Automatic checkpointing

**Usage:**

```bash
python examples/classification_training.py
```

**Expected Output:**

- Trained model checkpoint
- Training logs with loss/accuracy curves
- Best validation accuracy: ~92% on CIFAR-10

---

### 2. Object Detection Inference (`object_detection_inference.py`)

Run inference with trained YOLO or CenterNet models on images/videos.

**Features:**

- YOLO and CenterNet support
- Image and video processing
- Real-time webcam detection
- Bounding box visualization
- NMS post-processing

**Usage:**

```bash
# Process image
python examples/object_detection_inference.py \
    --model checkpoints/yolo_best.ckpt \
    --type yolo \
    --source image.jpg \
    --output ./results

# Process video
python examples/object_detection_inference.py \
    --model checkpoints/centernet.ckpt \
    --type centernet \
    --source video.mp4 \
    --display

# Webcam (real-time)
python examples/object_detection_inference.py \
    --model checkpoints/yolo_best.ckpt \
    --source 0 \
    --display
```

---

### 3. Tracking Device Benchmark (`tracking_device_benchmark.py`)

Benchmark tracking filters (Kalman, Particle) across different devices.

**Features:**

- Synthetic trajectory generation
- Multi-device testing (CPU, CUDA, MPS)
- Performance metrics (FPS, tracking error)
- Automatic device detection
- Device-specific recommendations

**Usage:**

```bash
# Full benchmark (all devices)
python examples/tracking_device_benchmark.py --frames 100

# Specific device
python examples/tracking_device_benchmark.py --device mps --frames 200

# Compare filters
python examples/tracking_device_benchmark.py --frames 50 --particles 1000
```

**Expected Output:**

```
============================================================
SUMMARY
============================================================
Filter                    Device     FPS        Avg Error
------------------------------------------------------------
Kalman                    mps        1011.9     1.25
Particle GPU (500)        mps        778.8      3.03
```

---

### 4. Audio Beamforming (`audio_beamforming.py`)

Direction-of-arrival (DOA) estimation and spatial audio filtering.

**Features:**

- Circular/linear microphone arrays
- Delay-and-sum beamforming
- MUSIC algorithm
- Spatial spectrum visualization
- Multi-source localization

**Usage:**

```bash
# DOA estimation
python examples/audio_beamforming.py \
    --mode doa \
    --input recording.wav \
    --n-mics 8 \
    --array-type circular

# Beamforming extraction
python examples/audio_beamforming.py \
    --mode beamform \
    --input recording.wav \
    --angle 45 \
    --method das \
    --output ./beamformed
```

**Requirements:**

- Multi-channel WAV file (8 channels for 8-mic array)
- `scipy` for audio processing
- `matplotlib` for visualization

---

### 5. Continual Learning (`continual_learning.py`)

Task-incremental learning with catastrophic forgetting prevention.

**Features:**

- Learning Without Forgetting (LWF)
- Task-incremental CIFAR-100 (5 tasks × 20 classes)
- Knowledge distillation
- Dynamic head expansion
- Forgetting metrics

**Usage:**

```bash
python examples/continual_learning.py
```

**Expected Output:**

```
📈 CONTINUAL LEARNING SUMMARY
Task     After T1  After T2  After T3  After T4  After T5
-------------------------------------------------------
Task 1      95.2%     92.1%     89.3%     87.5%     86.2%
Task 2                96.1%     93.4%     91.2%     89.8%
Task 3                          95.8%     93.7%     92.1%
Task 4                                    96.3%     94.5%
Task 5                                              95.9%

📊 Final Average Accuracy: 91.7%
🧠 Average Forgetting: 5.3%
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install APTT with dependencies
uv sync --extra cpu  # For CPU-only
# or
uv sync --extra cuda  # For CUDA support
```

### Running Examples

All examples are self-contained and include:

- ✅ Argument parsing for easy configuration
- ✅ Progress bars and logging
- ✅ Automatic device detection
- ✅ Comprehensive error messages

Simply run with Python:

```bash
python examples/<example_name>.py --help
```

---

## 📊 Device Support

| Example             | CPU | CUDA | MPS (Apple Silicon) | TPU |
| ------------------- | --- | ---- | ------------------- | --- |
| Classification      | ✅  | ✅   | ✅                  | ✅  |
| Detection Inference | ✅  | ✅   | ✅                  | ❌  |
| Tracking Benchmark  | ✅  | ✅   | ✅                  | ✅  |
| Audio Beamforming   | ✅  | ✅   | ✅                  | ✅  |
| Continual Learning  | ✅  | ✅   | ✅                  | ✅  |

All examples automatically detect and use the best available device.

---

## 🎯 Example Selection Guide

**I want to...**

- **Train a classifier**: → `classification_training.py`
- **Run detection on images/video**: → `object_detection_inference.py`
- **Compare tracking performance**: → `tracking_device_benchmark.py`
- **Process audio from mic arrays**: → `audio_beamforming.py`
- **Learn incrementally without forgetting**: → `continual_learning.py`

---

## 💡 Tips

1. **Start Simple**: Begin with `classification_training.py` to understand the basic workflow
2. **Device Auto-Detection**: All examples use `get_best_device()` - no manual configuration needed
3. **Logging**: Check `lightning_logs/` for TensorBoard logs after training
4. **Checkpoints**: Models are automatically saved to `checkpoints/`
5. **Visualization**: Most examples support `--output` flag for saving results

---

## 🐛 Troubleshooting

**Import Error: "No module named 'deepsuite'"**

```bash
# Install in editable mode
uv pip install -e .
```

**CUDA Out of Memory**

```bash
# Reduce batch size
python examples/classification_training.py --batch-size 64
```

**MPS Backend Error (Mac)**

```bash
# Fallback to CPU if MPS has issues
python examples/<example>.py --device cpu
```

---

## 📚 Further Reading

- [API Documentation](../docs/index.rst)
- [Tracking Filters Guide](../docs/tracking_filters.md)
- [Project Report](../PROJEKT_BERICHT.md)

---

## 🤝 Contributing

Add new examples? Follow this structure:

1. **Comprehensive docstring** at the top
2. **Argument parser** for configuration
3. **Clear console output** with emojis for visual structure
4. **Automatic device detection** via `get_best_device()`
5. **Error handling** with helpful messages
6. **Update this README** with usage instructions

Example template:

```python
"""
Example Title
=============

Description of what this example demonstrates.

Features:
- Feature 1
- Feature 2
"""

import argparse
from deepsuite.utils.device import get_best_device

def main():
    parser = argparse.ArgumentParser(description='...')
    # Add arguments
    args = parser.parse_args()

    device = args.device or str(get_best_device())
    print(f"🚀 Example Name")
    print(f"   Device: {device}")
    # Implementation

if __name__ == '__main__':
    main()
```

Happy experimenting! 🎉
