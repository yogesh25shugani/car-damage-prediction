
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


 
def build_model(
    num_classes=6,
    pretrained=True,
    dropout=0.3,
    fine_tune_layer3=False,
    fine_tune_layer4=True
):
    # Load pretrained ResNet50
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze Layer3 when requested
    if fine_tune_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # Unfreeze Layer4 when requested
    if fine_tune_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the original classifier
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model