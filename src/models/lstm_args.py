# Created by Hansi at 1/7/2022

import json
import os
from dataclasses import asdict, field, dataclass


@dataclass
class LSTMArgs:
    best_model_dir: str = "output/best_model"
    model_name: str = None
    cache_dir: str = "cache_dir/"

    early_stopping: bool = True
    early_stopping_min_delta: float = 0.0001
    early_stopping_patience: int = 5

    learning_rate: float = 1e-3  # 0.001

    manual_seed: int = None

    max_features: int = None
    max_len: int = 256
    embedding_size: int = None

    num_train_epochs: int = 20

    reduce_lr_on_plateau: bool = True
    reduce_lr_on_plateau_factor: float = 0.6
    reduce_lr_on_plateau_patience: int = 2  # 1
    reduce_lr_on_plateau_min_lr: float = 0.0001

    test_batch_size: int = 128
    train_batch_size: int = 128

    labels_list: list = field(default_factory=list)
    embedding_details: dict = field(default_factory=dict)

    train_file: str = 'train.tsv'
    dev_file: str = 'dev.tsv'

    not_saved_args: list = field(default_factory=list)

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {key: value for key, value in asdict(self).items() if key not in self.not_saved_args}
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(self.get_args_for_saving(), f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


