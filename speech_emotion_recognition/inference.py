import fire
import torch
from hydra import compose, initialize

from speech_emotion_recognition.core.classifier import AudioClassifier
from speech_emotion_recognition.core.model import EmotionSpeechClassifier
from speech_emotion_recognition.data.inference_data import AudioPredictDataModule


def predict(
    paths: str,
    ckpt_path: str = None,
    config_path: str = "../conf",
    config_name: str = "config",
):
    torch.set_float32_matmul_precision("medium")

    with initialize(config_path=config_path, version_base=None):
        config = compose(config_name=config_name)
    if ckpt_path:
        config.inference.ckpt_path = ckpt_path

    model = EmotionSpeechClassifier.load_from_checkpoint(
        config.inference.ckpt_path,
        model=AudioClassifier(config.model.num_classes),
        config=config,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dm = AudioPredictDataModule(config, paths.split())
    dm.setup("predict")
    dataloader = dm.predict_dataloader()

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())

    final_preds = torch.cat(all_preds)
    print(f"Predicted classes: {final_preds.tolist()}")


if __name__ == "__main__":
    fire.Fire(predict)
