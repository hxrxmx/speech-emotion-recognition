from pathlib import Path

import requests
from tqdm import tqdm

PUBLIC_URL = "https://disk.yandex.ru/d/mzf3S-CDwQssfw"
OUTPUT_PATH = Path("../../models/model-epoch=78-val_loss=0.7900-val_acc=0.674.ckpt")


def download_from_yadisk():
    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists.")
        return

    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": PUBLIC_URL}

    r = requests.get(api_url, params=params)
    r.raise_for_status()
    download_url = r.json()["href"]

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(OUTPUT_PATH, "wb") as file,
            tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading"
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(len(chunk))


if __name__ == "__main__":
    download_from_yadisk()
