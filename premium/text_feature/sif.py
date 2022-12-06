#!/usr/bin/env python
from typing import Any, List

import codefast as cf
import jieba
import numpy as np
from rich import print

from premium.utils import try_import

try_import('gensim')

from gensim.models import Word2Vec


def word2vec(texts: List[str]) -> Word2Vec:
    tokens_list = [jieba.lcut(_) for _ in texts]
    tokens_list = [list(filter(lambda x: len(x) > 1, _)) for _ in tokens_list]
    model = Word2Vec(tokens_list, window=5, min_count=1, workers=4)
    return model


def sif_embeddings(sentences: List[str], model: Word2Vec, alpha: float = 1e-3):
    """Compute the SIF embeddings for a list of sentences
    Parameters
    ----------
    sentences : list
        The sentences to compute the embeddings for
    model : `~gensim.models.base_any2vec.BaseAny2VecModel`
        A gensim model that contains the word vectors and the vocabulary
    alpha : float, optional
        Parameter which is used to weigh each individual word based on its probability p(w).
    Returns
    -------
    numpy.ndarray 
        SIF sentence embedding matrix of dim len(sentences) * dimension
    """
    REAL = np.float32
    if isinstance(model, Word2Vec):
        vlookup = model.wv.key_to_index  
        vectors = model.wv  
        size = model.vector_size
    else: # KeyedVectors
        vlookup = model.key_to_index  # Gives us access to word index and count
        vectors = model  # Gives us access to word vectors
        size = model.vector_size  # Embedding size

    Z = sum(vectors.get_vecattr(k, "count")
            for k in vlookup)  # Total word count
    output = []

    for s in sentences:
        tokens = [_ for _ in jieba.lcut(s) if _]
        count = 0
        v = np.zeros(size, dtype=REAL)  # Summary vector
        for w in tokens:
            if w in vlookup:
                v += (alpha /
                      (alpha +
                       (vectors.get_vecattr(w, 'count') / Z))) * vectors[w]
                count += 1

        output.append(v)
    return np.vstack(output).astype(REAL)


def top_k_similar_sentences(sentence: str,
                            corpus: List[str],
                            model: Word2Vec,
                            k: int = 10,
                            candidate_embedding=None) -> List[str]:
    """Find the top k similar sentences to a given sentence
    Parameters
    ----------
    sentence : str
        The sentence to find similar sentences for
    candidates : list
        The list of sentences to compare against
    model : `~gensim.models.base_any2vec.BaseAny2VecModel`
        A gensim model that contains the word vectors and the vocabulary
    k : int, optional
        The number of similar sentences to return
    Returns
    -------
    list
        The top k similar sentences
    """
    sentence_embedding = sif_embeddings([sentence], model)[0]
    if candidate_embedding is None:
        candidate_embedding = sif_embeddings(corpus, model)
    assert len(corpus) == len(
        candidate_embedding
    ), 'corpus and candidate_embedding must have the same length'
    norm = np.linalg.norm(candidate_embedding, axis=1) * np.linalg.norm(sentence_embedding)
    norm = np.where(norm == 0, 1, norm)
    cos_sim = np.dot(candidate_embedding, sentence_embedding) / norm
    top_k = np.argsort(cos_sim)[::-1][:k]
    return [(cos_sim[i], corpus[i]) for i in top_k]
