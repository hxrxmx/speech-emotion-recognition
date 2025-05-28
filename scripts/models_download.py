import requests


def download_from_yadisk(public_url, output_path):
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": public_url}

    r = requests.get(api_url, params=params)
    r.raise_for_status()
    download_url = r.json()["href"]

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


if __name__ == "__main__":
    download_from_yadisk(
        "https://disk.yandex.ru/d/IPrxAyV03h70HQ",
        "../models/model-epoch=90-val_loss=0.8091-val_acc=0.658.ckpt",
    )
