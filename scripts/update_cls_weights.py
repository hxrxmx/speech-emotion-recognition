import os

import hydra
from omegaconf import OmegaConf

conf_path = "../conf/"
conf_name = "config"

model_conf_path = "./conf/model"
model_conf_name = "model"


@hydra.main(config_path=conf_path, config_name=conf_name, version_base=None)
def main(config):
    ds_path = config.data.data_loading.train_data_path

    cls_to_count = {}
    for dir_name in os.listdir(ds_path):
        cls_to_count[dir_name] = len(os.listdir(os.path.join(ds_path, dir_name)))

    sum_items = sum(cls_to_count.values())

    cls_to_freq = {cls: count / sum_items for cls, count in cls_to_count.items()}

    OmegaConf.set_struct(config.model.weights, False)
    for cls, freq in cls_to_freq.items():
        config.model.weights[cls] = 1 / freq

    OmegaConf.save(config.model, f"{model_conf_path}/{model_conf_name}.yaml")


if __name__ == "__main__":
    main()
