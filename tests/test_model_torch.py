from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip('torch')

PROJECT_DIR = Path(__file__).resolve().parents[1] / 'Face Emotion Recognition'
sys.path.insert(0, str(PROJECT_DIR))

from deep_emotion_model import DeepEmotionModel
from emotion_detector import EmotionDetector, build_arg_parser


def test_model_forward_shape():
    model = DeepEmotionModel(num_classes=7)
    x = torch.randn(2, 1, 48, 48)
    out = model(x)
    assert out.shape == (2, 7)


def test_transform_contains_grayscale_and_normalize(tmp_path):
    with patch.object(EmotionDetector, 'load_model', return_value=MagicMock()):
        detector = EmotionDetector('model.pth', str(tmp_path))

    names = [t.__class__.__name__ for t in detector.transformation.transforms]
    assert 'Grayscale' in names
    assert 'Normalize' in names


def test_load_model_uses_map_location(tmp_path):
    model_file = tmp_path / 'model.pth'
    model_file.write_text('placeholder')

    dummy_model = MagicMock()
    with patch('emotion_detector.DeepEmotionModel', return_value=dummy_model), patch(
        'emotion_detector.torch.load', return_value={'weights': 'mock'}
    ) as mock_load:
        detector = EmotionDetector.__new__(EmotionDetector)
        detector.device = torch.device('cpu')
        detector.model_path = str(model_file)

        loaded_model = EmotionDetector.load_model(detector)

    assert loaded_model is dummy_model
    mock_load.assert_called_once_with(str(model_file), map_location=detector.device)


def test_cli_flag_test_acc_parses():
    parser = build_arg_parser()
    args = parser.parse_args(['--model', 'm.pth', '--data', './data', '--test_acc'])
    assert args.test_acc is True
