"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @created on: 03-03-2022 08:46:34
"""

import os
import joblib
from math import log, sqrt
from typing import List, Dict, Union, Tuple
from .tokenization import T_Tokenizer, tokenize_default


class BM25:
    """
    BM-25 implementation

    Params
    -------------------------
    corpus: List of tokenized documents.
        Default: None
    tokenizer: If None, then documents are assumed to be tokenized otherwise call
        this 'tokenizer' function on corpus before processing.
        Default: Spacy tokenizer
    k1: Controls tf saturation
        Default: 1.2
    k2: Controls idf saturation
        Default: 1.0
    b : Controls document-length normalization
        Default: 0.75
    """

    def __init__(
        self,
        corpus: Union[List[str], List[List[str]]] = None,
        tokenizer: T_Tokenizer = tokenize_default,
        k1: float = 1.2,
        k2: float = 1.0,
        b: float = 0.75,
    ) -> None:
        # Model hyperparams
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.tokenizer = tokenizer

        # Compute model
        self.tfs = []  # Term frequencies for each document: List[Dict[str,int]]
        self.dfs = {}  # Document frequency of each token: Dict[str,int]
        self.idfs = {}  # Inverse document frequency of each token: Dict[str,int]
        self.corpus_size = 0  # Corpus size
        self.avgdl = 0  # Average document length

        if corpus is not None:
            # Tokenize documents if needed
            if isinstance(corpus[0], str):
                if self.tokenizer is None:
                    raise ValueError(
                        "'corpus' is not tokenized, thus 'tokenizer' must be defined"
                    )
                corpus = self.tokenizer(corpus)

            # Extract basic parameters
            tfs, dfs, sum_doc_lens = self._extract_params(corpus)

            # Update state parameters
            self.tfs = tfs
            self.dfs = dfs
            self.corpus_size = len(self.tfs)
            self.avgdl = sum_doc_lens / self.corpus_size

            # Inverse document frequency
            if self.corpus_size > 0:
                for token, freq in self.dfs.items():
                    self.idfs[token] = self.get_idf(freq)

    def get_idf(self, doc_freq: float) -> float:
        """
        MOST IMPORTANT FUNCTION #1
        Just by modifying this function a little, one can implement other variations:
            - BM25-Okapi
            - BM25-Plus
            - BM25-L

        Params
        -------------------------
        doc_freq: Document frequency of any token in the corpus

        Returns
        -------------------------
        idf: Inverse document frequency
        """
        # same as log(k2 + (N+f-0.5)/(f+0.5))
        idf = log(
            self.corpus_size + doc_freq * (self.k2 - 1) + 0.5 * (self.k2 + 1)
        ) - log(doc_freq + 0.5)

        return idf

    def get_freq_imp(self, term_freq: float, doc_len: float) -> float:
        """
        MOST IMPORTANT FUNCTION #2
        Mathematically describes the term frequency part of BM25 equation

        Params
        -------------------------
        term_freq: Term frequency of any token in a document
        doc_len: Document length

        Returns
        -------------------------
        freq_imp: Frequency importance (It has no official name btw)
        """
        freq_imp = (
            term_freq
            * (self.k1 + 1)
            / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        )

        return freq_imp

    def _extract_params(
        self,
        corpus: List[List[str]],
    ) -> Tuple[List[Dict[str, int]], Dict[str, int], int]:
        """
        Extracts basic parameters from tokenized corpus, specifically
            - Term frequencies
            - Document frequencies
            - Sum of all document lengths
        """
        tfs = []
        dfs = {}
        sum_doc_lens = 0

        # Read a document one by one
        for document in corpus:
            if len(document) == 0:
                continue

            # Term frequencies
            doc_tf = {}
            for token in document:
                doc_tf[token] = doc_tf.get(token, 0) + 1
            tfs.append(doc_tf)

            # Document frequencies
            for token in doc_tf.keys():
                dfs[token] = dfs.get(token, 0) + 1

            # Update avgdl
            sum_doc_lens += len(document)

        return tfs, dfs, sum_doc_lens

    def resume(
        self,
        corpus: Union[List[str], List[List[str]]],
        tokenizer: T_Tokenizer = tokenize_default,
        save_path: str = None,
    ) -> None:
        """
        Resume/update current model on given corpus, retains previous documents as well

        Params
        -------------------------
        corpus: List of tokenized documents.
            Default: None
        tokenizer: If None, then documents are assumed to be tokenized otherwise call
            this 'tokenizer' function on corpus before processing.
            Default: Spacy tokenizer
        save_path: If not None, save model to save_path.
            Default: None
        """
        # Assert
        assert len(corpus) == 0, "Train corpus can't be empty"

        # Tokenize documents if needed
        if isinstance(corpus[0], str):
            if tokenizer is None:
                raise ValueError(
                    "'corpus' is not tokenized, thus 'tokenizer' must be defined"
                )
            corpus = tokenizer(corpus)

        # Train model
        tfs, dfs, sum_doc_lens = self._extract_params(corpus)

        # Update tfs
        self.tfs.extend(tfs)

        # Update dfs
        for token, freq in dfs.items():
            self.dfs[token] = self.dfs.get(token, 0) + freq

        # Update avgdl
        old_corpus_size = self.corpus_size
        self.corpus_size = len(self.tfs)
        self.avgdl = (
            self.avgdl * old_corpus_size / self.corpus_size
            + sum_doc_lens / self.corpus_size
        )

        # Update idfs
        updated_tokens = dfs.keys()
        for token in updated_tokens:
            self.idfs[token] = self.get_idf(self.dfs[token])

        # Save
        if save_path is not None:
            self.save(save_path=save_path)

    @staticmethod
    def load(model_path: str):
        """Staticmethod which loads and returns BM25 model"""
        # Load
        state_params = joblib.load(model_path)

        # Fill values
        model = BM25(
            k1=state_params["k1"],
            k2=state_params["k2"],
            b=state_params["b"],
        )
        model.tfs = state_params["tfs"]
        model.dfs = state_params["dfs"]
        model.idfs = state_params["idfs"]
        model.corpus_size = state_params["corpus_size"]
        model.avgdl = state_params["avgdl"]

        return model

    def save(self, save_path: str) -> None:
        """Save state parameters using joblib"""
        # Save all parameters
        state_params = {
            "k1": self.k1,
            "k2": self.k2,
            "b": self.b,
            "tfs": self.tfs,
            "dfs": self.dfs,
            "idfs": self.idfs,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
        }

        # Save
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(state_params, save_path)

    def bow_of(self, query: List[str]) -> Dict[str, float]:
        """
        Bag of words does not contain unseen words
        Params
        -------------------------
        query: Tokenized document

        Returns:
        bow: Bag of words of query
        """
        if isinstance(query, str):
            if self.tokenizer is None:
                raise ValueError("'tokenizer' must be defined if query is not tokenized.")
            query = self.tokenizer(query)
        bow = {}
        for token in query:
            if token in self.idfs:
                bow[token] = bow.get(token, 0) + 1

        for token in bow.keys():
            bow[token] = self.idfs[token] * self.get_freq_imp(bow[token], len(query))

        return bow

    def vectorize(self, query: List[str]) -> List[float]:
        """
        Vectorize query
        Params
        -------------------------
        query: Tokenized document

        Returns:
        vec: Vectorized query of same len
        """
        bow = self.bow_of(query)
        vec = [0.0] * len(query)

        for idx, token in enumerate(query):
            if token in bow:
                vec[idx] = bow[token]

        assert len(query) == len(vec)
        return vec

    def similarity(self, s1: List[str], s2: List[str]) -> float:
        """
        Compute cosine similarity
        """
        # Calculate bows
        bow1 = self.bow_of(s1)
        bow2 = self.bow_of(s2)
        all_known_tokens = set(bow1.keys()).union(set(bow2.keys()))

        xx = 0  # x^2
        yy = 0  # y^2
        p = 0  # dot product
        for token in all_known_tokens:
            x = bow1.get(token, 0)
            y = bow2.get(token, 0)
            p += x * y
            xx += x * x
            yy += y * y

        # cosine
        if xx and yy:
            res = p / sqrt(xx * yy)
        else:
            res = 0.0

        # Clip cosine-similarity between 0 and 1 (handles small floating point errors)
        res = max(0.0, min(res, 1.0))

        return res
