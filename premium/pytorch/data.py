from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split


class BaseDataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass


class TextDataset(BaseDataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Callable):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = row['label']
        tokens = self.tokenizer(text)[0]
        return {'text': tokens, 'label': label}


def train_test_val_split(df, test_size=0.2, val_size=0.2, random_state=42):
    X, P = train_test_split(df, test_size=test_size, random_state=random_state)
    V, T = train_test_split(P, test_size=val_size, random_state=random_state)
    return X, V, T


class AbstractLoader(ABC):
    def __init__(self) -> None:
        super().__init__()


from torch.utils.data import DataLoader


class TextLoader(AbstractLoader):
    def __init__(self, dataset, **kargs):
        self._DataSet = dataset
        self.kargs = kargs

    def __call__(self, df: pd.DataFrame, **kargs) -> DataLoader:
        self.kargs.update(kargs)
        return DataLoader(self._DataSet(df), **self.kargs)
