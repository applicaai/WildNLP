from abc import abstractmethod
from abc import ABC
import nltk
from pattern.en import conjugate as conjugate_en
from pattern.en import PRESENT, PROGRESSIVE, PAST, PARTICIPLE


class GrammarHelper(ABC):

    @abstractmethod
    def pos_tag(self, tokens):
        pass

    @abstractmethod
    def get_wordnet_pos(self, pos):
        pass

    @property
    @abstractmethod
    def pos_tags_to_omit(self):
        pass

    @staticmethod
    @abstractmethod
    def conjugate_verb(base_word, full_part_of_speech):
        pass


class NltkGrammarHelper(GrammarHelper):

    def __init__(self):
        nltk.download('averaged_perceptron_tagger', quiet=True)
        self._POS_NLTK_WORDNET_DICT = {'j': 'a'}
        # NNP - proper noun, singular ‘Harrison’, NNPS - proper noun, plural ‘Americans’
        self._POS_TAGS_TO_OMIT = {'NNP', 'NNPS'}

    @property
    def pos_tags_to_omit(self):
        return self._POS_TAGS_TO_OMIT

    def pos_tag(self, tokens):
        return nltk.pos_tag(tokens)

    def get_wordnet_pos(self, pos):
        short_pos = pos[0].lower()

        if short_pos in self._POS_NLTK_WORDNET_DICT:
            return self._POS_NLTK_WORDNET_DICT[short_pos]

        return short_pos

    @staticmethod
    def conjugate_verb(base_word, full_part_of_speech):
        # base form: go
        if full_part_of_speech == 'VB':
            new_word = base_word
        # past tense: went
        elif full_part_of_speech == 'VBD':
            new_word = conjugate_en(base_word, tense=PAST)
        # gerund/present participle: going
        elif full_part_of_speech == 'VBG':
            new_word = conjugate_en(base_word, tense=PRESENT, aspect=PARTICIPLE)
        # past participle: gone
        elif full_part_of_speech == 'VBN':
            new_word = conjugate_en(base_word, tense=PAST, aspect=PARTICIPLE)
        # singular present, non-3rd person: go
        elif full_part_of_speech == 'VBP':
            new_word = base_word
        # singular present, 3rd person: goes
        elif full_part_of_speech == 'VBZ':
            new_word = conjugate_en(base_word, tense=PRESENT, person=3)
        else:
            print("Wrong part of speech tag")
            new_word = base_word

        return new_word



