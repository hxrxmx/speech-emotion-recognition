import os
import random
import shutil
from collections import defaultdict

DATA_DIR = "data/CREMA-D/AudioWAV"
SPLIT_DIR = "data/CREMA-D-split"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 4423


def split_dataset():
    random.seed(SEED)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(SPLIT_DIR, split), exist_ok=True)

    class_dict = defaultdict(list)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    for f in files:
        class_name = f.split("_")[2]
        class_dict[class_name].append(f)

    for class_name, file_list in class_dict.items():
        random.shuffle(file_list)

        train_end = int(len(file_list) * TRAIN_RATIO)
        val_end = train_end + int(len(file_list) * VAL_RATIO)

        train_files = file_list[:train_end]
        val_files = file_list[train_end:val_end]
        test_files = file_list[val_end:]

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(SPLIT_DIR, split, class_name), exist_ok=True)

        for file, split in zip(
            [train_files, val_files, test_files],
            ["train", "val", "test"],
        ):
            for f in file:
                shutil.copy(
                    os.path.join(DATA_DIR, f),
                    os.path.join(SPLIT_DIR, split, class_name, f),
                )

    print("Dataset splitted with stratifications:;)")


if __name__ == "__main__":
    split_dataset()
