import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from deep_emotion_model import DeepEmotionModel


class EmotionDetector:
    def __init__(self, model_path, data_path, use_webcam=False, test_accuracy=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.data_path = data_path
        self.use_webcam = use_webcam
        self.test_accuracy = test_accuracy
        self.model = self.load_model()
        self.transformation = self.get_transformations()
        self.classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def load_model(self):
        model = DeepEmotionModel()
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_transformations(self):
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def _build_test_dataset(self):
        dataset_path = Path(self.data_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Test data path not found: {dataset_path}")

        return datasets.ImageFolder(root=str(dataset_path), transform=self.transformation)

    def evaluate_test_data(self):
        test_dataset = self._build_test_dataset()
        test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = (correct / total) * 100 if total else 0.0
        print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def webcam_emotion_detection(self):
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        print("Webcam mode started. Press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (48, 48))
            tensor = self.transformation(face).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                pred = int(logits.argmax(dim=1).item())

            label = self.classes[pred]
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Emotion detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.test_accuracy:
            self.evaluate_test_data()
        if self.use_webcam:
            self.webcam_emotion_detection()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Emotion Detection")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--test_acc", action="store_true", help="Evaluate test accuracy")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    detector = EmotionDetector(args.model, args.data, args.webcam, args.test_acc)
    detector.run()
