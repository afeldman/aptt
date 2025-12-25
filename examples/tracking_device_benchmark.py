"""Benchmark tracking filters across different devices.

This script compares tracking performance on CPU, CUDA, and MPS (Apple Silicon).
Automatically detects available devices and runs benchmarks.

Usage:
    python examples/tracking_device_benchmark.py

    # Or specify device:
    python examples/tracking_device_benchmark.py --device mps
"""

import argparse
import time

import torch

from deepsuite.tracker.tracker import KalmanFilter, ParticleFilter, ParticleFilterGPU
from deepsuite.utils.device import get_best_device


def generate_synthetic_trajectory(num_frames: int = 100) -> list[list[float]]:
    """Generate a synthetic moving bounding box trajectory.

    Args:
        num_frames: Number of frames to generate

    Returns:
        List of bounding boxes [x1, y1, x2, y2]
    """
    trajectory = []
    x, y = 100.0, 100.0
    vx, vy = 2.0, 1.5

    for _i in range(num_frames):
        # Add some randomness
        x += vx + (torch.randn(1).item() * 0.5)
        y += vy + (torch.randn(1).item() * 0.5)

        # Bounce off walls
        if x < 50 or x > 450:
            vx *= -1
        if y < 50 or y > 350:
            vy *= -1

        w, h = 100.0, 80.0
        trajectory.append([x, y, x + w, y + h])

    return trajectory


def benchmark_filter(filter_class, filter_name: str, device: str, num_frames: int = 100, **kwargs):
    """Benchmark a tracking filter.

    Args:
        filter_class: Filter class to instantiate
        filter_name: Name for display
        device: Device to run on
        num_frames: Number of frames to process
        **kwargs: Additional arguments for filter

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {filter_name} on {device}")
    print(f"{'=' * 60}")

    # Generate trajectory
    trajectory = generate_synthetic_trajectory(num_frames)
    initial_box = trajectory[0]

    # Initialize filter
    try:
        filter_obj = filter_class(initial_box, device=device, **kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return None

    # Warm-up
    for _ in range(5):
        filter_obj.predict()
        filter_obj.update(trajectory[0])

    # Benchmark
    start_time = time.time()

    predictions = []
    for _i, obs_box in enumerate(trajectory):
        pred_box = filter_obj.predict()
        predictions.append(
            pred_box.cpu().detach().numpy() if torch.is_tensor(pred_box) else pred_box
        )
        filter_obj.update(obs_box)

    # Synchronize for accurate GPU timing
    if device in ["cuda", "mps"]:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time

    # Compute tracking error (simplified)
    total_error = 0.0
    for pred, gt in zip(predictions, trajectory, strict=False):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        error = sum(abs(p - g) for p, g in zip(pred[:2], gt[:2], strict=False))  # Center error
        total_error += error

    avg_error = total_error / num_frames

    results = {
        "filter": filter_name,
        "device": device,
        "frames": num_frames,
        "time": elapsed_time,
        "fps": fps,
        "avg_error": avg_error,
    }

    print(f"✅ Completed in {elapsed_time:.3f}s")
    print(f"   FPS: {fps:.1f}")
    print(f"   Avg Error: {avg_error:.2f} pixels")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tracking filters")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu, or auto)"
    )
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to process")
    parser.add_argument(
        "--all-devices", action="store_true", help="Benchmark on all available devices"
    )
    args = parser.parse_args()

    print("🚀 APTT Tracking Filter Benchmark")
    print("=" * 60)

    # Detect available devices
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available_devices.append("mps")

    print(f"Available devices: {available_devices}")

    if args.device:
        devices_to_test = [args.device]
    elif args.all_devices:
        devices_to_test = available_devices
    else:
        # Auto-detect best device
        best_device = str(get_best_device())
        devices_to_test = [best_device]
        print(f"Auto-detected best device: {best_device}")

    # Define filters to benchmark
    filters_to_test = [
        ("Kalman", KalmanFilter, {}),
        ("Particle (100)", ParticleFilter, {"num_particles": 100}),
    ]

    # Add GPU filters only for GPU devices
    if any(d in devices_to_test for d in ["cuda", "mps"]):
        filters_to_test.extend(
            [
                ("Particle GPU (500)", ParticleFilterGPU, {"num_particles": 500}),
                ("Particle GPU (1000)", ParticleFilterGPU, {"num_particles": 1000}),
            ]
        )

    # Run benchmarks
    all_results = []

    for device in devices_to_test:
        for filter_name, filter_class, kwargs in filters_to_test:
            # Skip GPU filters on CPU
            if device == "cpu" and "GPU" in filter_name:
                continue

            result = benchmark_filter(filter_class, filter_name, device, args.frames, **kwargs)

            if result:
                all_results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Filter':<25} {'Device':<10} {'FPS':<10} {'Avg Error':<12} {'Time (s)':<10}")
    print("-" * 80)

    for r in all_results:
        print(
            f"{r['filter']:<25} {r['device']:<10} {r['fps']:<10.1f} "
            f"{r['avg_error']:<12.2f} {r['time']:<10.3f}"
        )

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if "mps" in available_devices:
        print("✅ MPS (Apple Silicon) detected!")
        print("   → Use tracker_type='particle_gpu' for best performance")
        print("   → Example: TrackingModule(model, tracker_type='particle_gpu')")

    if "cuda" in available_devices:
        print("✅ CUDA (NVIDIA GPU) detected!")
        print("   → Use tracker_type='particle_gpu' for best performance")

    if len(available_devices) == 1 and available_devices[0] == "cpu":
        print("ℹ️  CPU-only detected")
        print("   → Use tracker_type='kalman' for best speed")
        print("   → Or tracker_type='particle' for better accuracy (slower)")

    print("\n💡 Tip: The framework auto-detects the best device by default!")
    print("   Just use: TrackingModule(model, tracker_type='particle_gpu')")


if __name__ == "__main__":
    main()
