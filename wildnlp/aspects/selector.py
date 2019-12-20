import numpy as np
import copy
from abc import abstractmethod
from abc import ABC
from flair.data import Sentence
from flair.training_utils import clear_embeddings
import teapot


class Selector(ABC):
    """Abstract for classes that chooses word that best fits the sentence from list of words."""

    @abstractmethod
    def select_best_token(self, tokens, new_token_candidates, replaced_token_ind):
        pass

    @staticmethod
    def _similarity(v1, v2):
        """
        Measure cosine similarity between two vectors.

        :param v1: list of float numbers
        :param v2: list of float numbers

        :return: similarity between v1 and v2, float from -1 to 1
        """
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        return float(np.dot(v1, v2) / n1 / n2)


class EmbeddingSelector(Selector):

    def __init__(self, embedding):
        self._embedding = embedding

    def select_best_token(self, tokens, new_token_candidates, replaced_token_ind):
        """
        Chooses word that best fits the sentence from list of words using embedding from flair.

        :param tokens: tokens for words in sentence, list of str
        :param new_token_candidates: tokens for new word candidates, list of str
        :param replaced_token_ind: index of word to replace

        :return: word that best fits the sentence
        """
        sentence = Sentence(" ".join(tokens))
        sentences = []
        tokens_copy = copy.deepcopy(tokens)
        for word in new_token_candidates:
            tokens_copy[replaced_token_ind] = word
            sentences.append(Sentence(" ".join(tokens_copy)))

        self._embedding.embed(sentences + [sentence])
        similarities = []
        org_word_emb = sentence.tokens[replaced_token_ind].embedding

        for changed_sent in sentences:
            swap_word_emb = changed_sent.tokens[replaced_token_ind].embedding
            similarities.append(self._similarity(org_word_emb, swap_word_emb))

        clear_embeddings(sentences + [sentence])

        return new_token_candidates[np.argmax(similarities)]


class TeapotSelector(Selector):

    def __init__(self, scorer="chrf", path_to_meteor_jar=None):
        """

        :param scorer: scorer from teapot: chrf, meteor or bleu (default: chrf)
        :param path_to_meteor_jar: path METEOR to jar file (http://www.cs.cmu.edu/~alavie/METEOR/)
        """

        if scorer == "chrf":
            self._scorer = teapot.ChrF()
        elif scorer == "meteor":
            self._scorer = teapot.METEOR(path_to_meteor_jar, java_command="java -Xmx2G -jar")
        elif scorer == "bleu":
            self._scorer = teapot.BLEU()

    def select_best_token(self, tokens, new_token_candidates, replaced_token_ind):
        """
        Chooses word that best fits the sentence from list of words using teapotNLP scorer.

        :param tokens: tokens for words in sentence, list of str
        :param new_token_candidates: tokens for new word candidates, list of str
        :param replaced_token_ind: index of word to replace

        :return: word that best fits the sentence
        """
        sentence = Sentence(" ".join(tokens))
        sentences = []
        tokens_copy = copy.deepcopy(tokens)
        for word in new_token_candidates:
            tokens_copy[replaced_token_ind] = word
            sentences.append(Sentence(" ".join(tokens_copy)))

        similarities = []

        for changed_sent in sentences:
            similarity = self._scorer.score([changed_sent.to_plain_string()],
                                            [sentence.to_plain_string()], lang='english')[0]
            similarities.append(similarity)

        return new_token_candidates[np.argmax(similarities)]
