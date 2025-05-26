import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def download_data():
    dataset_name = "ejlok1/cremad"
    download_path = Path("../data/CREMA-D")
    zip_path = Path("../data/cremad.zip")

    if download_path.exists():
        print(f"{download_path} already exists. Skipping download.")
        return

    download_path.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset_name, path=download_path.parent, unzip=False)

    print("Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_path)

    print(f"Dataset extracted to {download_path}")

    zip_path.unlink()


if __name__ == "__main__":
    download_data()
