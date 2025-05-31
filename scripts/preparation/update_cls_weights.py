from pathlib import Path

import hydra
from omegaconf import OmegaConf

MODEL_CONF_PATH = Path("../conf/model")
MODEL_CONF_NAME = "model"


@hydra.main(config_path="../conf/", config_name="config", version_base=None)
def main(config):
    ds_path = Path(config.data.data_loading.train_data_path)

    cls_to_count = {
        class_dir.name: len(list(class_dir.iterdir()))
        for class_dir in ds_path.iterdir()
        if class_dir.is_dir()
    }

    sum_items = sum(cls_to_count.values())

    cls_to_freq = {cls: count / sum_items for cls, count in cls_to_count.items()}

    OmegaConf.set_struct(config.model.weights, False)
    for cls, freq in cls_to_freq.items():
        config.model.weights[cls] = 1 / freq

    MODEL_CONF_PATH.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config.model, MODEL_CONF_PATH / f"{MODEL_CONF_NAME}.yaml")

    print("success")


if __name__ == "__main__":
    main()
