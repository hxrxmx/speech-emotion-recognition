import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

DATA_DIR = Path("../data/CREMA-D/AudioWAV")
SPLIT_DIR = Path("../data/CREMA-D-split")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
MIN_SAMPLE_TIME = 0.2


def is_valid_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    duration = waveform.shape[1] / sr
    if duration < MIN_SAMPLE_TIME:
        return False
    if torch.isnan(waveform).any():
        return False
    return True


def split_dataset():
    random.seed(SEED)

    class_dict = defaultdict(list)

    files = sorted([file for file in DATA_DIR.glob("*.wav") if is_valid_audio(file)])
    for file in files:
        parts = file.name.split("_")
        class_name = parts[2]
        class_dict[class_name].append(file)

    for split in ["train", "val", "test"]:
        (SPLIT_DIR / split).mkdir(parents=True, exist_ok=True)

    for class_name, file_list in class_dict.items():
        random.shuffle(file_list)

        n_total = len(file_list)
        train_end = int(n_total * TRAIN_RATIO)
        val_end = train_end + int(n_total * VAL_RATIO)

        splits = {
            "train": file_list[:train_end],
            "val": file_list[train_end:val_end],
            "test": file_list[val_end:],
        }

        for split_name, split_files in splits.items():
            target_dir = SPLIT_DIR / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in split_files:
                shutil.copy(file_path, target_dir / file_path.name)

        print(
            f"class: {class_name},",
            f"train files: {len(splits['train'])},",
            f"val files: {len(splits['val'])},",
            f"test files: {len(splits['test'])}",
        )

    print("Dataset splitted:;)")


if __name__ == "__main__":
    split_dataset()
