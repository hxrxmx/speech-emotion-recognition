import torch

from speech_emotion_recognition.classifier import AudioClassifier


def test_model_prediction():
    model = AudioClassifier(num_classes=6)
    random_input = torch.rand(1, 1, 256, 1024)

    pred = model(random_input)
    assert pred.shape == (1, 6), "output should be in form (batch_size, num_classes=6)!"
    assert torch.allclose(
        pred.sum(), torch.tensor(1.0)
    ), "predictions should sum to 1.!"
