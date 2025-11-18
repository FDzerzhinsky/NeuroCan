import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

from config import cfg
from data import get_transforms
from models import ResNetCanClassifier


class CanRotationPredictor:
    """Класс для предсказания угла поворота банки"""

    def __init__(self, model_path):
        self.device = cfg.DEVICE
        self.transform = get_transforms('val')

        self.model = ResNetCanClassifier(num_classes=cfg.NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Best accuracy in training: {checkpoint.get('best_accuracy', 'N/A')}%")

    def predict(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0].item()

        return {
            'angle': predicted_class.item(),
            'confidence': confidence,
            'all_probabilities': probabilities.cpu().numpy()[0]
        }

    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['file_path'] = str(image_path)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict soda can rotation angle')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image or directory for prediction')
    args = parser.parse_args()

    predictor = CanRotationPredictor(args.model)
    input_path = Path(args.image)

    if input_path.is_file():
        result = predictor.predict(input_path)
        print(f"Image: {input_path}")
        print(f"Predicted angle: {result['angle']}°")
        print(f"Confidence: {result['confidence']:.4f}")

    elif input_path.is_dir():
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        results = predictor.predict_batch(image_paths)

        print(f"Processed {len(results)} images:")
        for result in results:
            print(f"  {Path(result['file_path']).name}: "
                  f"{result['angle']}° (conf: {result['confidence']:.4f})")

    else:
        print(f"Error: {args.image} not found!")


if __name__ == '__main__':
    main()