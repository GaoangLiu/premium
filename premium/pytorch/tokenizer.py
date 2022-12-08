#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import List, Union
import torch
import torchtext as tt
import numpy as np 

class Vectorizer(ABC):
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


class VocabVectorizer(Vectorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tt.data.get_tokenizer('basic_english')
        self.vocab = None
        self.max_length = kwargs.get('max_length', 100)
        self.padding = kwargs.get('padding', False)

    def _yield_tokens(self, texts: List[str]):
        for text in texts:
            yield self.tokenizer(text)

    def __len__(self):
        return len(self.vocab)

    def size(self)->int:
        return len(self.vocab)
    
    def fit(self, texts: List[str]) -> 'Self':
        self.vocab = tt.vocab.build_vocab_from_iterator(self._yield_tokens(texts))
        self.vocab.set_default_index(0)
        return self

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors = [[self.vocab[token]
                    for token in self.tokenizer(text)[:self.max_length]]
                   for text in texts]
        padded_vectors = np.array([
            vector + [0] * (self.max_length - len(vector)) for vector in vectors
        ])
        return padded_vectors


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


