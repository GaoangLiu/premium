#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import List, Union
import torch
import torchtext as tt

class Tokenizer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, texts: List[str]):
        pass

    @abstractmethod
    def transform(self, texts: List[str]):
        pass

    @abstractmethod
    def fit_transform(self, texts: List[str]):
        pass

    @abstractmethod
    def inverse_transform(self, tokens: List[int]):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class VocabTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tt.data.get_tokenizer('basic_english')
        self.vocab = None
        self.max_size = kwargs.get('max_size', 10000)
        # tt.vocab.Vocab(self.tokenizer(yield_tokens(self.df, 'text')), max_size=Configs.MAX_WORDS)

    def _yield_tokens(self, texts: List[str]):
        for text in texts:
            yield self.tokenizer(text)

    def __len__(self):
        return len(self.vocab)

    def fit(self, texts: List[str]) -> 'Self':
        assert isinstance(texts, list), "texts must be a list"
        self.vocab = tt.vocab.Vocab(self._yield_tokens(texts), max_size=self.max_size)
        self.vocab.set_default_index(0)
        return self

    def transform(self, texts: Union[str, List[str]]) -> List[List[int]]:
        if isinstance(texts, str):
            texts = [texts]
        return [[self.vocab[tok] for tok in self.tokenizer(text)] for text in texts]

    def fit_transform(self, texts: Union[str, List[str]]) -> List[List[int]]:
        if isinstance(texts, str):
            texts = [texts]
        return self.fit(texts).transform(texts)

    def inverse_transform(self, tokens: List[int]) -> List[str]:
        return [self.vocab.get_itos[tok] for tok in tokens]

    def save(self, path: str) -> 'Self':
        torch.save(self.vocab, path)
        return self

    def load(self, path: str) -> 'Self':
        self.vocab = torch.load(path)
        return self

    def __call__(self, texts: Union[str, List[str]]) -> List[int]:
        return self.transform(texts)
