import pytest
from omegaconf import OmegaConf

from speech_emotion_recognition.data.data import CREMADataModule


@pytest.fixture
def datamodule(tmp_path):
    test_config = {
        "data": {
            "data_loading": {
                "train_data_path": "data/CREMA-D-split/train",
                "val_data_path": "data/CREMA-D-split/val",
                "test_data_path": "data/CREMA-D-split/test",
            },
            "augmentations": {
                "min_gain_db": -6.0,
                "max_gain_db": 12.0,
                "min_shift": -0.1,
                "max_shift": 0.5,
                "min_pitch_shift_st": -2,
                "max_pitch_shift_st": 3,
                "hp_max_cutoff_freq": 800,
                "lp_min_cutoff_freq": 4000,
            },
            "preprocessing": {
                "sample_rate": 16000,
                "n_mels": 256,
                "max_sample_time": 5.0,
                "n_fft": 1024,
                "win_length": 768,
                "hop_length": 64,
                "f_min": 60.0,
                "f_max": 8000.0,
                "top_db": 60.0,
                "noise_offset_db": 7.0,
                "n_target_time_frames": 1024,
            },
        },
        "training": {
            "batch_size": 2,
            "lr": 1e-4,
            "num_workers": 1,
            "num_epochs": 100,
            "sheduler": {
                "factor": 1.0,
                "patience": 1,
            },
        },
        "model": {
            "num_classes": 6,
            "weights": {
                "ANG": 5.85531496062992,
                "DIS": 5.85531496062992,
                "FEA": 5.85531496062992,
                "HAP": 5.85531496062992,
                "NEU": 6.845799769850403,
                "SAD": 5.85531496062992,
            },
        },
    }

    conf = OmegaConf.create(test_config)
    dm = CREMADataModule(config=conf)
    dm.setup("fit")
    return dm


@pytest.mark.requires_files
def test_data_modules(datamodule):
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    spectrogram, labels = batch
    assert spectrogram.shape[0] == 2, "first shape should be batch_size"
    assert set(labels.numpy()) <= {0, 1, 2, 3, 4, 5}, "Class should be from 0 to 5!"
