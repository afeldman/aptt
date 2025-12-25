"""Object Detection Inference Example.
===================================

This example demonstrates how to use a trained object detection model
for inference on images or video streams.

Features:
- Load trained YOLO/CenterNet models
- Process images and videos
- Visualize detections with bounding boxes
- Export results to JSON/CSV
- Real-time webcam detection
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from deepsuite.model.detection.centernet import CenterNetModel
from deepsuite.model.detection.yolo import YOLOv5
from deepsuite.utils.bbox import nms
from deepsuite.utils.device import get_best_device


class DetectionInference:
    """Object detection inference pipeline."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "yolo",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or str(get_best_device())

        print(f"🔧 Loading {model_type} model from {model_path}")
        print(f"   Device: {self.device}")

        # Load model
        if model_type == "yolo":
            self.model = YOLOv5.load_from_checkpoint(model_path)
        elif model_type == "centernet":
            self.model = CenterNetModel.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded successfully!")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to model input size (typically 640x640)
        img_size = 640
        h, w = image.shape[:2]
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to square
        pad_h = (img_size - new_h) // 2
        pad_w = (img_size - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_h,
            img_size - new_h - pad_h,
            pad_w,
            img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Convert to tensor
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, scale, (pad_w, pad_h)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on image.

        Returns:
            boxes: (N, 4) array of [x1, y1, x2, y2]
            scores: (N,) array of confidence scores
            classes: (N,) array of class indices
        """
        # Preprocess
        img_tensor, scale, (pad_w, pad_h) = self.preprocess(image)

        # Inference
        outputs = self.model(img_tensor)

        # Post-process (simplified - actual implementation depends on model)
        # This is a generic example
        if isinstance(outputs, dict):
            boxes = outputs.get("boxes", outputs.get("pred_boxes"))
            scores = outputs.get("scores", outputs.get("pred_scores"))
            classes = outputs.get("classes", outputs.get("pred_classes"))
        else:
            # Assume outputs is [boxes, scores, classes]
            boxes, scores, classes = outputs[:3]

        # Convert to numpy
        boxes = boxes.cpu().numpy()[0]
        scores = scores.cpu().numpy()[0]
        classes = classes.cpu().numpy()[0]

        # Filter by confidence
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # NMS
        keep = nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        # Rescale boxes to original image size
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        return boxes, scores, classes

    def visualize(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        class_names: list[str] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes on image."""
        vis_image = image.copy()

        for box, score, cls in zip(boxes, scores, classes, strict=False):
            x1, y1, x2, y2 = box.astype(int)

            # Color based on class
            color = tuple(int(c) for c in np.random.RandomState(int(cls)).randint(0, 255, 3))

            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_names[int(cls)] if class_names else int(cls)}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                vis_image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1
            )
            cv2.putText(
                vis_image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        return vis_image


def process_image(args) -> None:
    """Process single image."""
    detector = DetectionInference(
        model_path=args.model,
        model_type=args.type,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Load image
    image = cv2.imread(args.source)
    if image is None:
        raise ValueError(f"Failed to load image: {args.source}")

    print(f"\n📸 Processing image: {args.source}")
    print(f"   Size: {image.shape[1]}x{image.shape[0]}")

    # Detect
    boxes, scores, classes = detector.predict(image)
    print(f"   Detected {len(boxes)} objects")

    # Visualize
    vis_image = detector.visualize(image, boxes, scores, classes)

    # Save
    output_path = Path(args.output) / f"{Path(args.source).stem}_detected.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)

    print(f"✅ Saved to {output_path}")


def process_video(args) -> None:
    """Process video file or webcam."""
    detector = DetectionInference(
        model_path=args.model,
        model_type=args.type,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Open video
    if args.source == "0":
        cap = cv2.VideoCapture(0)
        print("\n📹 Opening webcam...")
    else:
        cap = cv2.VideoCapture(args.source)
        print(f"\n📹 Processing video: {args.source}")

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {args.source}")

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolution: {width}x{height} @ {fps} FPS")

    # Setup output video
    output_path = Path(args.output) / f"{Path(args.source).stem}_detected.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            boxes, scores, classes = detector.predict(frame)

            # Visualize
            vis_frame = detector.visualize(frame, boxes, scores, classes)

            # Write
            out.write(vis_frame)

            # Display
            if args.display:
                cv2.imshow("Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   Processed {frame_count} frames...", end="\r")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\n✅ Processed {frame_count} frames, saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="APTT Object Detection Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--type", type=str, default="yolo", choices=["yolo", "centernet"], help="Model type"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Image/video path or 0 for webcam"
    )
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (auto-detect if not specified)"
    )
    parser.add_argument("--display", action="store_true", help="Display video in real-time")

    args = parser.parse_args()

    print("🎯 APTT Object Detection Inference")
    print("=" * 60)

    # Determine if source is image or video
    source_path = Path(args.source)
    if args.source == "0" or source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        process_video(args)
    else:
        process_image(args)


if __name__ == "__main__":
    main()
