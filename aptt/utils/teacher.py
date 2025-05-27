# apu.ml/continual/teacher_utils.py

import copy

import torch
import torch.nn as nn


def copy_teacher(student_model: nn.Module) -> nn.Module:
    """
    Erstellt eine Kopie des aktuellen Modells als Teacher.

    Die Kopie wird eval()-gesetzt und eingefroren (no gradient updates).
    """
    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    freeze_teacher(teacher_model)
    return teacher_model


def freeze_teacher(model: nn.Module):
    """
    Friert alle Parameter eines Modells ein (kein Gradient mehr).
    """
    for param in model.parameters():
        param.requires_grad = False


def load_teacher(checkpoint_path: str, model_class, **model_kwargs) -> nn.Module:
    """
    Sicheres Laden eines gespeicherten Teacher-Modells.
    NUR State Dicts werden akzeptiert.
    """
    model = model_class(**model_kwargs)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        # Lightning Checkpoint
        state_dict = state_dict["state_dict"]

    # Sicherstellen, dass nur die Parameter geladen werden
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys or unexpected_keys:
        print(f"Warnung: Missing keys {missing_keys}, Unexpected keys {unexpected_keys}")

    model.eval()
    freeze_teacher(model)
    return model
