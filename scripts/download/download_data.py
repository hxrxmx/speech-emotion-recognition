import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_NAME = "ejlok1/cremad"
DOWNLOAD_PATH = Path("../../data/CREMA-D")
ZIP_PATH = Path("../../data/cremad.zip")


def download_data():
    if DOWNLOAD_PATH.exists():
        print(f"{DOWNLOAD_PATH} already exists.")
        return

    DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_PATH.parent, unzip=False)

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_PATH)

    print(f"Dataset extracted to {DOWNLOAD_PATH}")

    ZIP_PATH.unlink()


if __name__ == "__main__":
    download_data()
