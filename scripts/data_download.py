import os
import zipfile

import requests

DATASET_URL = (
    "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip"
)
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "crema-d.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "CREMA-D")


def download_dataset():
    response = requests.get(DATASET_URL, stream=True)
    with open(ZIP_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def unpack_dataset():
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(EXTRACT_PATH)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading...")
        download_dataset()
        print("Success")
    else:
        print("Zip_file already exists")

    if not os.path.exists(EXTRACT_PATH):
        print("Unpacking...")
        unpack_dataset()
        print("All done!")
    else:
        print("Unpacked file already exists!")


if __name__ == "__main__":
    main()
