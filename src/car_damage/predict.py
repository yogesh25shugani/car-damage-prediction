
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image, UnidentifiedImageError

from .model import build_model
from .data import get_transforms


# Class names must follow the same order used during training
CLASS_NAMES = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal"
]


def load_trained_model(checkpoint_path, device=None):
    """
    Load the trained ResNet50 model from a checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Recreate the same architecture used in Experiment 2
    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        dropout=0.3,
        fine_tune_layer3=False,
        fine_tune_layer4=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, device



def predict_image(image_input, model, device=None):
    """
    Predict car damage from an image path or PIL Image.
    """

    if device is None:
        device = next(model.parameters()).device

    # Load image depending on input type
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")

    else:
        image_path = Path(image_input)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(
                f"Unable to process image: {image_path}"
            ) from error

    # Apply evaluation transformations
    _, eval_transform = get_transforms(image_size=224)

    image_tensor = eval_transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(
        probabilities, dim=1
    )

    predicted_class = CLASS_NAMES[predicted_idx.item()]

    class_probabilities = {
        CLASS_NAMES[index]: round(
            probabilities[0][index].item(), 4
        )
        for index in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "class_probabilities": class_probabilities
    }

