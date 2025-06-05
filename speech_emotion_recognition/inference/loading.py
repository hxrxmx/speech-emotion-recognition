import torch

from speech_emotion_recognition.core.classifier import AudioClassifier


def load_model(ckpt_path, device, num_classes=6):
    model = AudioClassifier(num_classes=num_classes)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model
