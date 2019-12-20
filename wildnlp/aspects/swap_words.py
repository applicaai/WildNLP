import random
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from wildnlp.aspects.word_creator import HypernymCreator
from wildnlp.aspects.grammar_helper import NltkGrammarHelper
from wildnlp.aspects.base import Aspect
import copy
from flair.data import Sentence, Token


class SwapWords(Aspect):
    """Swap words for synonyms or hypernyms.

    .. caution:: Uses random numbers, default seed is 42.
    """

    def __init__(self, word_creator=None, words_percentage=50, words_number=None, pos_to_change=None,
                 selector=None, lemmatizer=None, stop_words=None,
                 grammar_helper=None, seed=42):
        """


        :param word_creator: Object that has function get_words which takes as arguments:
            word and part of speech in wordnet format and returns list of words.
            (default .word_creator.SynonymCreator)

        :param words_percentage: Percentage of words in a
            sentence that should be transformed. If greater than 0,
            always at least single word will be transformed.

        :param pos_to_change: List of parts of speech to change in wordnet format.
            v - verb, r - adverb, a - adjective, n - noun (default None)

        :param selector: Object that has function select_best_word which takes as arguments:
            list of all words, candidates to swap word, index of word to swap and returns new word.
            (default None)

        :param lemmatizer: Object that has function lemmatize which takes as arguments:
            word and part of speech in wordnet format and return lemma of that word.
            (default nltk.stem.wordnet.WordNetLemmatizer)

        :param stop_words: List of words that will not be swapped. (default nltk.corpus.stopwords.words('english'))

        :param grammar_helper: Object that has method pos_tag which takes as arguments:
            list of tokens and return list of tuples (word, tag in grammar_helper format),
            method get_wordnet_pos which takes as argument part of speech in grammar_helper format
            and returns part of speech in wordnet format, method conjugate_verb which takes as arguments word in base
            form and part of speech in grammar_helper format and returns conjugated word,  property pos_tags_to_omit
            which returns list of part of speech in grammar_helper format tags of words that will not be swapped
            (default .pos_tagger.NltkGrammarHelper)

        :param seed: Random seed.
        """
        if word_creator is None:
            self._word_creator = HypernymCreator()
        else:
            self._word_creator = word_creator

        if words_percentage >= 1:
            words_percentage /= 100.

        self._words_percentage = words_percentage

        self._words_number = words_number

        self._pos_to_change = pos_to_change

        self._selector = selector

        if lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        else:
            self._lemmatizer = lemmatizer

        if stop_words is None:
            nltk.download('stopwords', quiet=True)
            self._stop_words = stopwords.words('english')
        else:
            self._stop_words = stop_words

        if grammar_helper is None:
            self._grammar_helper = NltkGrammarHelper()
        else:
            self._grammar_helper = grammar_helper

        self.seed = seed

    def __call__(self, sentence):
        # tokens = self._tokenize(sentence)
        sentence_object = Sentence(sentence, True)
        sentence_object.tokens = [token for token in sentence_object.tokens if token.text != ""]
        tokens = [token.text for token in sentence_object.tokens]
        modified_tokens = self._swap_words(tokens)

        for org_token, new_token_text in zip(sentence_object, modified_tokens):
            if org_token.text != new_token_text:
                org_token.text = new_token_text

        # return self._detokenize(modified_tokens)
        return sentence_object.to_plain_string()

    def _swap_words(self, tokens):

        random.seed(self.seed)

        modified = copy.deepcopy(tokens)

        word_pos_list = self._grammar_helper.pos_tag(tokens)

        filtered_tokens = self._filter_tokens_by_pos(tokens, word_pos_list)
        num_words_to_change = self._percentage_to_num(filtered_tokens, self._words_percentage)
        if self._words_number is not None:
            num_words_to_change = self._words_number
        tokens_len = len(tokens)
        for i in random.sample(range(tokens_len), tokens_len):

            if num_words_to_change == 0:
                break

            token = word_pos_list[i][0]

            full_part_of_speech = word_pos_list[i][1]

            if full_part_of_speech in self._grammar_helper.pos_tags_to_omit:
                continue

            part_of_speech = self._grammar_helper.get_wordnet_pos(full_part_of_speech)

            if (self._pos_to_change is not None and part_of_speech not in self._pos_to_change) or \
                    token in self._stop_words:
                continue

            new_tokens = self._word_creator.get_words(token, part_of_speech)

            if len(new_tokens) == 0:
                continue

            if self._selector is None:
                new_token = random.choice(new_tokens)
            else:
                new_token = self._selector.select_best_token(tokens, new_tokens, i)

            if part_of_speech == 'v':
                new_token = self._grammar_helper.conjugate_verb(self._lemmatizer.lemmatize(new_token, part_of_speech),
                                                                full_part_of_speech)

            # Change new token first letter to upper case
            if token[0].isupper():
                new_token = new_token.title()

            modified[i] = new_token
            num_words_to_change -= 1

        return modified

    def _same_lemma(self, word1, word2, part_of_speech):
        return self._lemmatizer.lemmatize(word1, part_of_speech) == self._lemmatizer.lemmatize(word2, part_of_speech)

    def _filter_tokens_by_pos(self, tokens, word_pos_list):
        if self._pos_to_change is None:
            filtered_tokens = tokens
        else:
            filtered_tokens = [word for word, pos in word_pos_list
                               if self._grammar_helper.get_wordnet_pos(pos) in self._pos_to_change]
        return filtered_tokens

    @staticmethod
    def _percentage_to_num(array, percentage):
        if percentage == 0:
            return 0
        # Ensure that at least one item will be transformed.
        return max(1, int(len(array) * percentage))
