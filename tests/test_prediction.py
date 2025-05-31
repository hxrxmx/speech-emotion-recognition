import torch

from speech_emotion_recognition.core.classifier import AudioClassifier


def test_model_prediction():
    model = AudioClassifier(num_classes=6)
    model.eval()
    random_input = torch.rand(1, 1, 256, 1024)

    pred = model(random_input)
    assert pred.shape == (1, 6), "output should be in form (batch_size, num_classes=6)!"
    assert torch.allclose(
        pred.sum(), torch.tensor(1.0)
    ), "predictions should sum to 1.!"
