import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

from config.config import cfg
from data.transforms import get_transforms
from models.resnet_model import ResNetCanClassifier


class CanRotationPredictor:
    """Класс для предсказания угла поворота банки с поддержкой grayscale"""

    def __init__(self, model_path):
        self.device = cfg.DEVICE
        self.transform = get_transforms('val')

        # Загружаем модель
        self.model = ResNetCanClassifier(num_classes=cfg.NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Best accuracy in training: {checkpoint.get('best_accuracy', 'N/A')}%")
        print(f"Inference mode: {'Grayscale' if cfg.GRAYSCALE else 'RGB'}")

    def predict(self, image_path):
        """Предсказывает угол для одного изображения"""
        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Преобразуем в правильное цветовое пространство
        if cfg.GRAYSCALE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)  # Добавляем канальное измерение
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Применяем трансформы
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Предсказание
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
        """Предсказывает углы для батча изображений"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['file_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict soda can rotation angle')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image or directory for prediction')
    args = parser.parse_args()

    # Инициализируем predictor
    predictor = CanRotationPredictor(args.model)
    input_path = Path(args.image)

    if input_path.is_file():
        # Предсказание для одного файла
        result = predictor.predict(input_path)
        print(f"Image: {input_path}")
        print(f"Predicted angle: {result['angle']}°")
        print(f"Confidence: {result['confidence']:.4f}")

    elif input_path.is_dir():
        # Предсказание для всех изображений в папке
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
        if not image_paths:
            print(f"No images found in {input_path}")
            return

        results = predictor.predict_batch(image_paths)

        print(f"Processed {len(results)} images:")
        for result in results:
            print(f"  {Path(result['file_path']).name}: "
                  f"{result['angle']}° (conf: {result['confidence']:.4f})")

    else:
        print(f"Error: {args.image} not found!")


if __name__ == '__main__':
    main()