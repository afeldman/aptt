from apu.ml.loss.bbox import BboxLoss
from apu.ml.loss.detection import DetectionLoss
from apu.ml.loss.dfl import DFLoss
from apu.ml.loss.distill import Distill
from apu.ml.loss.focal import BinaryFocalLoss, FocalLoss, MulticlassFocalLoss
from apu.ml.loss.keypoint import KeypointLoss
from apu.ml.loss.lwf import LwF
from apu.ml.loss.mel import MelLoss
from apu.ml.loss.rmse import RMSE
from apu.ml.loss.segmentation import SegmentationLoss
from apu.ml.loss.snr import (
    ScaleInvariantSignal2DistortionRatio,
    ScaleInvariantSignal2NoiseRatio,
    Signal2NoiseRatio,
)
from apu.ml.loss.varifocal import VarifocalLoss


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
    }
    if name not in loss_map:
        raise ValueError(f"Loss '{name}' nicht gefunden. Verfügbare: {list(loss_map.keys())}")
    return loss_map[name](**kwargs)
