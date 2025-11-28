import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import onnxruntime as ort

from config.config import cfg
from data.transforms import get_transforms
from models.resnet_model import ResNetCanClassifier


class CanRotationPredictor:
    """Класс для предсказания угла поворота банки с поддержкой grayscale и ONNX"""

    def __init__(self, model_path):
        self.device = cfg.DEVICE
        self.transform = get_transforms('val')
        self.model_path = Path(model_path)

        # Определяем тип модели по расширению файла
        if self.model_path.suffix.lower() == '.onnx':
            self.model_type = 'onnx'
            self._init_onnx_model()
        else:
            self.model_type = 'pytorch'
            self._init_pytorch_model()

        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Model type: {self.model_type.upper()}")
        print(f"✅ Inference mode: {'Grayscale' if cfg.GRAYSCALE else 'RGB'}")

    def _init_pytorch_model(self):
        """Инициализация PyTorch модели"""
        self.model = ResNetCanClassifier(num_classes=cfg.NUM_CLASSES)
        checkpoint = torch.load(self.model_path, map_location='cpu')

        # Проверяем структуру checkpoint
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Если checkpoint содержит сразу state_dict модели
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        if 'best_accuracy' in checkpoint:
            print(f"✅ Best accuracy in training: {checkpoint['best_accuracy']}%")

    def _init_onnx_model(self):
        """Инициализация ONNX модели"""
        # Создаем сессию ONNX Runtime
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider'] + providers

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )

        # Получаем информацию о входе модели
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✅ ONNX Input: {self.input_name}, Output: {self.output_name}")
        print(f"✅ ONNX Providers: {providers}")

    def predict_from_tensor(self, image_tensor):
        """Предсказание из готового тензора (для верификации)"""
        if self.model_type == 'pytorch':
            return self._predict_pytorch(image_tensor)
        else:
            return self._predict_onnx(image_tensor)

    def _predict_pytorch(self, image_tensor):
        """Предсказание с использованием PyTorch модели"""
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0].item()

        return {
            'angle': predicted_class.item(),
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'raw_output': outputs.cpu().numpy()[0]
        }

    def _predict_onnx(self, image_tensor):
        """Предсказание с использованием ONNX модели"""
        # Преобразуем тензор в numpy array
        input_data = image_tensor.numpy().astype(np.float32)

        # Выполняем инференс
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        outputs_tensor = torch.from_numpy(outputs[0])

        # Применяем softmax и получаем предсказания
        probabilities = torch.softmax(outputs_tensor, dim=1)
        predicted_class = torch.argmax(outputs_tensor, dim=1)
        confidence = torch.max(probabilities, dim=1)[0].item()

        return {
            'angle': predicted_class.item(),
            'confidence': confidence,
            'probabilities': probabilities.numpy()[0],
            'raw_output': outputs[0][0]
        }

    def predict(self, image_path):
        """Предсказывает угол для одного изображения из файла"""
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
        image_tensor = transformed['image'].unsqueeze(0)

        return self.predict_from_tensor(image_tensor)

    def predict_batch(self, image_paths):
        """Предсказывает углы для батча изображений"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['file_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict soda can rotation angle')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pth or .onnx)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image or directory for prediction')
    args = parser.parse_args()

    # Проверяем существование файла модели
    if not Path(args.model).exists():
        print(f"❌ Error: Model file {args.model} not found!")
        return

    # Инициализируем predictor
    predictor = CanRotationPredictor(args.model)
    input_path = Path(args.image)

    if input_path.is_file():
        # Предсказание для одного файла
        result = predictor.predict(input_path)
        print(f"📊 Image: {input_path}")
        print(f"🎯 Predicted angle: {result['angle']}°")
        print(f"📈 Confidence: {result['confidence']:.4f}")

    elif input_path.is_dir():
        # Предсказание для всех изображений в папке
        image_extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(ext))
            image_paths.extend(input_path.glob(ext.upper()))

        if not image_paths:
            print(f"❌ No images found in {input_path}")
            return

        results = predictor.predict_batch(image_paths)

        print(f"📊 Processed {len(results)} images:")
        for result in results:
            print(f"  📁 {Path(result['file_path']).name}: "
                  f"{result['angle']}° (conf: {result['confidence']:.4f})")

    else:
        print(f"❌ Error: {args.image} not found!")


if __name__ == '__main__':
    main()