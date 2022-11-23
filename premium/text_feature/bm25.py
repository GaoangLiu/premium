#!/usr/bin/env python

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class BM25(object):
    """
    Best Match 25.

    Parameters
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    corpus_ : list[list[str]]
        The input corpus.

    corpus_size_ : int
        Number of documents in the corpus.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query: List[str]) -> List[float]:
        scores = [
            self._score(query, index) for index in range(self.corpus_size_)
        ]
        return scores

    def _score(self, query: List[str], index: int) -> float:
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score


if __name__ == '__main__':
    # we'll generate some fake texts to experiment with
    corpus = [
        'Human machine interface for lab abc computer applications',
        'A survey of user opinion of computer system response time',
        'The EPS user interface management system',
        'System and human system engineering testing of EPS',
        'Relation of user perceived response time to error measurement',
        'The generation of random binary unordered trees',
        'The intersection graph of paths in trees',
        'Graph minors IV Widths of trees and well quasi ordering',
        'Graph minors A survey'
    ]

    # remove stop words and tokenize them (we probably want to do some more
    # preprocessing with our text in a real world setting, but we'll keep
    # it simple here)
    stopwords = set(['for', 'a', 'of', 'the', 'and', 'to', 'in'])
    texts = [[
        word for word in document.lower().split() if word not in stopwords
    ] for document in corpus]

    # build a word count dictionary so we can remove words that appear only once
    word_count_dict = {}
    for text in texts:
        for token in text:
            word_count = word_count_dict.get(token, 0) + 1
            word_count_dict[token] = word_count

    texts = [[token for token in text if word_count_dict[token] > 1]
             for text in texts]

    print(texts)

    # query our corpus to see which document is more relevant
    query = 'The intersection of graph survey and trees'
    query = [word for word in query.lower().split() if word not in stopwords]

    bm25 = BM25()
    bm25.fit(texts)
    scores = bm25.search(query)

    for score, doc in zip(scores, corpus):
        score = round(score, 3)
        print(str(score) + '\t' + doc)
