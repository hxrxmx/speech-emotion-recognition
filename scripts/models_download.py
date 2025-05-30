import requests
from tqdm import tqdm


def download_from_yadisk(public_url, output_path):
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": public_url}

    r = requests.get(api_url, params=params)
    r.raise_for_status()
    download_url = r.json()["href"]

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(output_path, "wb") as file,
            tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading"
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(len(chunk))


if __name__ == "__main__":
    download_from_yadisk(
        "https://disk.yandex.ru/d/IPrxAyV03h70HQ",
        "../models/model-epoch=90-val_loss=0.8091-val_acc=0.658.ckpt",
    )
