from pathlib import Path

import fire
import torch
from hydra import compose, initialize

from speech_emotion_recognition.core.classifier import AudioClassifier
from speech_emotion_recognition.core.model import EmotionSpeechClassifier


def prepare_model_input(config, device):
    rand_waveform = torch.randn(
        4,
        1,
        config.data.preprocessing.n_mels,
        config.data.preprocessing.n_target_time_frames,
    ).to(device)
    return rand_waveform


def convert_to_onnx(full_model, onnx_path, model_input):
    torch.onnx.export(
        model=full_model,
        args=model_input,
        f=onnx_path.as_posix(),
        input_names=["spectrogram"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"spectrogram": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def convert_model(
    ckpt_path,
    onnx_path=None,
    trt_path=None,
    config_path: str = "../../conf",
    config_name: str = "config",
):
    if not onnx_path:
        onnx_path = Path(ckpt_path).parent / (Path(ckpt_path).stem + ".onnx")
    if not trt_path:
        trt_path = Path(ckpt_path).parent / (Path(ckpt_path).stem + ".pth")

    ckpt_path = Path(ckpt_path)
    onnx_path = Path(onnx_path)
    trt_path = Path(trt_path)

    if onnx_path.exists():
        print(f"{onnx_path} already exists.")
    if trt_path.exists():
        print(f"{trt_path} already exists.")
    if onnx_path.exists() and trt_path.exists():
        return

    with initialize(config_path=config_path, version_base=None):
        config = compose(config_name=config_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionSpeechClassifier.load_from_checkpoint(
        ckpt_path,
        model=AudioClassifier(config.model.num_classes),
        config=config,
    ).model
    model.eval()
    model.to(device)

    model_input = prepare_model_input(config, device)

    if not onnx_path.exists():
        print("Export ONNX...")
        convert_to_onnx(model, onnx_path, model_input)
        print(f"ONNX saved in {onnx_path}")

    if not trt_path.exists():
        print("Export TorchScript (for TensorRT)...")
        traced = torch.jit.script(model, model_input)

        test_output = traced(model_input)
        print(f"Traced model output shape: {test_output.shape}")

        traced.save(trt_path.as_posix())
        print(f"TorchScript saved in {trt_path}")


if __name__ == "__main__":
    fire.Fire(convert_model)
