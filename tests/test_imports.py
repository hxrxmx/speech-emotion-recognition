def test_imports():
    from speech_emotion_recognition.classifier import AudioClassifier
    from speech_emotion_recognition.data import CREMADataModule
    from speech_emotion_recognition.loss import FocalLoss
    from speech_emotion_recognition.model import EmotionSpeechClassifier
    from speech_emotion_recognition.preprocessing import (
        AudioAugmentationsPipeline,
        MelSpecPreprocessingPipeline,
        WaveformPreprocessingPipeline,
    )

    CREMADataModule, EmotionSpeechClassifier, AudioClassifier, FocalLoss
    AudioAugmentationsPipeline, MelSpecPreprocessingPipeline
    WaveformPreprocessingPipeline

    assert True, "there're issues with imports"
