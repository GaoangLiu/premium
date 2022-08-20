from dataclasses import dataclass
from typing import Dict, List

import codefast as cf
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


@dataclass
class ModelConfig(object):
    epochs: int = 100
    batch_size: int = 8
    max_len: int = 64
    patience: int = 10
    model_path: str = "./model.h5"
    model_name: str = "model"
    model_type: str = "cnn"
    n_splits: int = 10

    def update(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def to_dict(self):
        return self.__dict__


class KerasCallbacks(object):
    def __init__(self, patience: int = 10, log_dir: str = "/tmp/logs") -> None:
        self.patience = patience
        self.log_dir = log_dir
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=True,
        )
        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.7, patience=patience // 2, verbose=1
        )
        self.tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=1,
        )
        self.csv_logger = CSVLogger(
            filename=f"{self.log_dir}/keraslog.csv", append=True, separator=";"
        )

        self.checkpoint = ModelCheckpoint(
            filepath=f"{self.log_dir}/keras.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

    def _validate_roc_mode(self, callbacks) -> bool:
        """Make sure monitor consistence with mode
        """
        for cb in callbacks:
            if cb.__dict__.get("monitor") == "val_auc":
                assert cb.mode == "max", "roc_mode must be max"
        return True

    @property
    def all(self) -> List[Dict]:
        # return all callbacks
        _callbacks = [v for _, v in self.__dict__.items() if isinstance(v, Callback)]
        names = [c.__class__.__name__ for c in _callbacks]
        cf.info("using callbacks:", names)
        return _callbacks

    def some(self, keys: List[str]) -> List[Dict]:
        # return only some callbacks
        callbacks = [self.__dict__[k] for k in keys]
        cf.info("using callbacks", list(map(lambda x: x.__class__.__name__, callbacks)))
        return callbacks
