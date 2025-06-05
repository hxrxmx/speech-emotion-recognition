def test_imports():
    from speech_emotion_recognition.core.classifier import AudioClassifier
    from speech_emotion_recognition.core.loss import FocalLoss
    from speech_emotion_recognition.core.model import EmotionSpeechClassifier
    from speech_emotion_recognition.data.augmentations import AudioAugmentationsPipeline
    from speech_emotion_recognition.data.data import CREMADataModule, CREMADataset
    from speech_emotion_recognition.data.preprocessing import (
        MelSpecPreprocessingPipeline,
        WaveformPreprocessingPipeline,
    )
    from speech_emotion_recognition.inference.loading import load_model
    from speech_emotion_recognition.inference.preprocessing import (
        InferencePreprocessing,
    )
    from speech_emotion_recognition.utils.plotting import LocalPlot

    assert callable(AudioClassifier)
    assert callable(FocalLoss)
    assert callable(EmotionSpeechClassifier)

    assert callable(AudioAugmentationsPipeline)
    assert callable(CREMADataModule)
    assert callable(CREMADataset)
    assert callable(MelSpecPreprocessingPipeline)
    assert callable(WaveformPreprocessingPipeline)

    assert callable(load_model)
    assert callable(InferencePreprocessing)

    assert callable(LocalPlot)
