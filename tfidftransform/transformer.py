from typing import List, Optional, Dict

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidftransform.util import to_lower


class Transformer:
    def __init__(self, stop_words: List[str] = None, language: str = None, convert_to_lower=False):
        """
        Responsible for TF-IDF (Term Frequency - Inverse Document Frequency)
        transformations by training and inference on texts.
        Creates Transformer object.

        :param stop_words: Words that are excluded from vocabulary and TF-IDF calculation.
        :param language: The language, can be 'turkish', 'english' etc.
        :param convert_to_lower: Option for converting to lowercase.
        """
        self._stop_words = stop_words
        self._language = language
        self._convert_to_lower = convert_to_lower
        self._model: Optional[TfidfVectorizer] = None

    def train(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        """
        Calculates TF-IDF vectors based on provided texts.

        :param texts: Texts used for TF-IDF calculations using their vocabulary.
        :return: A sparse TF-IDF matrix of size (num_samples, vocab_size)
        """
        self._check_texts(texts)

        texts = self._preprocess_texts(texts)

        self._model = TfidfVectorizer(stop_words=self._stop_words, lowercase=False)

        return self._model.fit_transform(texts)

    def infer(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        """
        Calculates TF-IDF matrix of provided texts based on trained vocabulary.

        :param texts: Texts used for TF-IDF calculations using trained vocabulary.
        :return: A sparse TF-IDF matrix of size (num_samples, vocab_size)
        """
        self._check_texts(texts)
        self._check_model()

        texts = self._preprocess_texts(texts)

        return self._model.transform(texts)

    def get_vocabulary(self) -> Dict[str, int]:
        """
        Returns the vocabulary and corresponding indices of training.

        :return: A dictionary of word-index pairs
        """
        self._check_model()
        return self._model.vocabulary_

    def _check_model(self):
        if self._model is None:
            raise ValueError("Training is not done yet")

    def _preprocess_texts(self, texts: List[str]):
        if self._convert_to_lower:
            return [to_lower(t, self._language) for t in texts]
        return texts

    @staticmethod
    def _check_texts(texts: List[str]):
        if texts is None:
            raise ValueError('Texts null')
        if not texts:
            raise ValueError('Texts empty')
