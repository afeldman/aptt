from aptt.loss.bbox import BboxLoss
from aptt.loss.detection import DetectionLoss
from aptt.loss.dfl import DFLoss
from aptt.loss.distill import Distill
from aptt.loss.focal import BinaryFocalLoss, FocalLoss, MulticlassFocalLoss
from aptt.loss.keypoint import KeypointLoss
from aptt.loss.lwf import LwF
from aptt.loss.mel import MelLoss
from aptt.loss.rmse import RMSE
from aptt.loss.segmentation import SegmentationLoss
from aptt.loss.snr import (
    ScaleInvariantSignal2DistortionRatio,
    ScaleInvariantSignal2NoiseRatio,
    Signal2NoiseRatio,
)
from aptt.loss.varifocal import VarifocalLoss
from aptt.loss.centernet import CenterNetLoss
from aptt.loss.heat import KeypointHeatmapLoss


def get_loss(name: str, **kwargs):
    loss_map = {
        "bbox": BboxLoss,
        "keypoint": KeypointLoss,
        "focal": FocalLoss,
        "binaryfocal": BinaryFocalLoss,
        "multiclassfocal": MulticlassFocalLoss,
        "detection": DetectionLoss,
        "distill": Distill,
        "dfl": DFLoss,
        "lwf": LwF,
        "mel": MelLoss,
        "rmse": RMSE,
        "segmentation": SegmentationLoss,
        "sdr": Signal2NoiseRatio,
        "si_snr": ScaleInvariantSignal2NoiseRatio,
        "si_sdr": ScaleInvariantSignal2DistortionRatio,
        "varfocal": VarifocalLoss,
        "centernet": CenterNetLoss,
        "keypointhearmap": KeypointHeatmapLoss
    }
    if name not in loss_map:
        raise ValueError(f"Loss '{name}' nicht gefunden. Verfügbare: {list(loss_map.keys())}")
    return loss_map[name](**kwargs)
