import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def visualize_prediction(image_path, predicted_angle, true_angle=None, confidence=None):
    image = plt.imread(image_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(image)

    title = f"Predicted: {predicted_angle}°"
    if true_angle is not None:
        title += f" | True: {true_angle}°"
    if confidence is not None:
        title += f" | Confidence: {confidence:.3f}"

    ax.set_title(title)
    ax.axis('off')

    return fig


def calculate_angular_error(predicted, true):
    error = abs(predicted - true)
    return min(error, 360 - error)


def load_model(checkpoint_path, num_classes=360):
    from models.resnet_model import ResNetCanClassifier
    model = ResNetCanClassifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model