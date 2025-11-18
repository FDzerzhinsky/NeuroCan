import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Базовая модель с циклической функцией потерь"""

    def __init__(self, num_classes=360):
        super().__init__()
        self.num_classes = num_classes

    def cyclic_loss(self, pred_angles, true_angles):
        pred_rad = pred_angles * 2 * torch.pi / 360
        true_rad = true_angles * 2 * torch.pi / 360

        pred_sin = torch.sin(pred_rad)
        pred_cos = torch.cos(pred_rad)
        true_sin = torch.sin(true_rad)
        true_cos = torch.cos(true_rad)

        sin_loss = torch.nn.functional.mse_loss(pred_sin, true_sin)
        cos_loss = torch.nn.functional.mse_loss(pred_cos, true_cos)

        return (sin_loss + cos_loss) / 2

    def angle_to_coordinates(self, angles):
        angles_rad = angles * 2 * torch.pi / 360
        sin = torch.sin(angles_rad)
        cos = torch.cos(angles_rad)
        return torch.stack([sin, cos], dim=1)

    def coordinates_to_angle(self, coordinates):
        sin, cos = coordinates[:, 0], coordinates[:, 1]
        angles_rad = torch.atan2(sin, cos)
        angles_deg = angles_rad * 360 / (2 * torch.pi)
        angles_deg = angles_deg % 360
        return angles_deg