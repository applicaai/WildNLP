from abc import abstractmethod
from abc import ABC
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from wildnlp.aspects.grammar_helper import NltkGrammarHelper
import gensim


class WordCreator(ABC):
    """Abstract for classes that  creates list of words for given word"""

    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()

    @abstractmethod
    def get_words(self, word, part_of_spech):
        pass

    def _same_lemma(self, word1, word2, part_of_speech):
        """Check if two words have same lemmas.

        :param word1: first word to compare
        :param word2: second word to compare
        :param part_of_speech: part_of_speech of words to compare

        :return: True if words have the same lemmas, False otherwise
        """
        return self._lemmatizer.lemmatize(word1, part_of_speech) == self._lemmatizer.lemmatize(word2, part_of_speech)


class SynonymCreator(WordCreator):

    def __init__(self):
        WordCreator.__init__(self)

    def get_words(self, word, part_of_speech):
        """
        For given word find synonyms in WordNet.

        :param word: word
        :param part_of_speech: part of speech to which word corresponding
        :return: list of synonyms of word from WordNet
        """
        synsets = wordnet.synsets(word)

        if len(synsets) == 0:
            return []

        # Save synonyms to list.
        synsets_names = []
        for synset in synsets:
            if synset.pos() != part_of_speech:
                continue
            synsets_names += synset.lemma_names()

        # Check if synonyms are different than word and synonym is not multipart.
        names = [name for name in synsets_names if name.lower() != word.lower()
                 and not self._same_lemma(name.lower(), word.lower(), part_of_speech)
                 and '_' not in name]

        # Remove duplicates.
        names = list(set(names))

        return names


class HypernymCreator(WordCreator):

    def __init__(self):
        WordCreator.__init__(self)

    def get_words(self, word, part_of_speech):
        """
        For given word find hypernyms in WordNet.

        :param word: word
        :param part_of_speech: part of speech to which word corresponding
        :return: list of hypernyms of word from WordNet
        """
        synsets = wordnet.synsets(word)

        if len(synsets) == 0:
            return []

        # Save synonyms to list.
        hypernyms_names = []
        for synset in synsets:
            if synset.pos() != part_of_speech:
                continue
            for hypernym in synset.hypernyms():
                hypernyms_names += hypernym.lemma_names()

        # Check if synonyms are different than word and synonym is not multipart.
        names = [name for name in hypernyms_names if name.lower() != word.lower()
                 and not self._same_lemma(name.lower(), word.lower(), part_of_speech)
                 and '_' not in name]

        # Remove duplicates.
        names = list(set(names))

        return names


class Word2VecCreator(WordCreator):

    def __init__(self, path_to_word2vec_file, n_most_similar=10, grammar_helper=None):
        """

        :param path_to_word2vec_file: path to file with embeddings in word2vec format
        :param n_most_similar: number of nearest neighbors
        :param grammar_helper: Object that has method pos_tag which takes as arguments:
            list of tokens and return list of tuples (word, tag in grammar_helper format),
            method get_wordnet_pos which takes as argument part of speech in grammar_helper format
            and returns part of speech in wordnet format, method conjugate_verb which takes as arguments word in base
            form and part of speech in grammar_helper format and returns conjugated word,  property pos_tags_to_omit
            which returns list of part of speech in grammar_helper format tags of words that will not be swapped
            (default .pos_tagger.NltkGrammarHelper)
        """
        WordCreator.__init__(self)
        self._glove = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec_file)
        self._n_most_similar = n_most_similar
        if grammar_helper is None:
            self._grammar_helper = NltkGrammarHelper()
        else:
            self._grammar_helper = grammar_helper

    def get_words(self, word, part_of_speech):
        """
        For given word find neighbors in Word2Vec.

        :param word: word
        :param part_of_speech: part of speech to which word corresponding
        :return: list of neighbors in Word2Vec
        """
        if word not in self._glove.vocab:
            return []

        most_similar = self._glove.most_similar(word.lower(), topn=self._n_most_similar*20)
        candidate_words = [new_word for new_word, score in most_similar
                           if not self._same_lemma(word.lower(), new_word.lower(), part_of_speech)
                           and self._same_pos(new_word.lower(), part_of_speech)]

        return candidate_words[:self._n_most_similar]

    def _same_pos(self, word, part_of_speech):
        """
        Check if given word corresponding to part of speech.

        :param word: word
        :param part_of_speech: part of speech
        :return: True if word corresponding to part of speech, False otherwise
        """
        word_pos_list = self._grammar_helper.pos_tag([word])
        full_part_of_speach = word_pos_list[0][1]
        if part_of_speech == self._grammar_helper.get_wordnet_pos(full_part_of_speach):
            return True
        return False






