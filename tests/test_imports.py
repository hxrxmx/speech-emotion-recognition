def test_imports():
    from speech_emotion_recognition.core.classifier import AudioClassifier
    from speech_emotion_recognition.core.loss import FocalLoss
    from speech_emotion_recognition.core.model import EmotionSpeechClassifier
    from speech_emotion_recognition.data.data import CREMADataModule
    from speech_emotion_recognition.data.inference_data import AudioPredictDataModule
    from speech_emotion_recognition.data.preprocessing import (
        AudioAugmentationsPipeline,
        MelSpecPreprocessingPipeline,
        WaveformPreprocessingPipeline,
    )

    assert callable(AudioClassifier)
    assert callable(FocalLoss)
    assert callable(EmotionSpeechClassifier)
    assert callable(CREMADataModule)
    assert callable(AudioPredictDataModule)
    assert callable(AudioAugmentationsPipeline)
    assert callable(MelSpecPreprocessingPipeline)
    assert callable(WaveformPreprocessingPipeline)
