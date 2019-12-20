from wildnlp.aspects.utils import compose
from wildnlp.aspects import *
from wildnlp.datasets import SampleDataset
from wildnlp.aspects.selector import EmbeddingSelector, TeapotSelector
from wildnlp.aspects.word_creator import HypernymCreator, SynonymCreator, Word2VecCreator
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, BertEmbeddings, ELMoEmbeddings
import argparse


corruptor_choices = ["Articles", "Digits2Words", "Misspellings", "Punctuation", "QWERTY", "RemoveChar", "SentimentMasking",
                     "Swap", "ChangeChar", "WhiteSpaces", "AddSubString", "SwapWords"]
word_creator_choices = ["synonyms", "hypernyms", "word2vec"]
selector_choices = ["random", 'flair_embedding', 'bert_embedding', 'elmo_embedding', 'chrf_teapot', 'bleu_teapot']
pos_choices = ["v", "r", "a", "n"]

parser = argparse.ArgumentParser()
parser.add_argument("--corruptors", help="corrupt functions", type=str, nargs="*", choices=corruptor_choices)
parser.add_argument("--path_to_tsv_file", help="path to tsv file with texts to corrupt", type=str)
parser.add_argument("--path_to_output_file", help="path to output file", type=str)
parser.add_argument("--text", help="text to corrupt", type=str)
parser.add_argument("--words_percentage", help="percentage of words to corrupt (RemoveChar, Swap, QWERTY, SwapWords)",
                    type=int)
parser.add_argument("--characters_percentage", help="percentage of characters to corrupt (RemoveChar, QWERTY)",
                    type=int)
parser.add_argument("--word_creator", help="word creator for SwapWords (default: hypernyms)", type=str,
                    choices=word_creator_choices)
parser.add_argument("--path_to_word2vec_file", help="path to embeddings in word2vec format file", type=str)
parser.add_argument("--selector", help="selector for SwapWords (default: random)", type=str, choices=selector_choices)
parser.add_argument("--pos", help="part of speech to change by SwapWords (v - verb, r - adverb, a - adjective,"
                                  " n - noun )", type=str, nargs="*", choices=pos_choices)
parser.add_argument("--char_to_mask_words", help="A character that will be used to mask word (SentimentMasking)",
                    type=str)
parser.add_argument("--punctuation_char", help="Punctuation mark that will be removed or added to sentences "
                                               "(Punctuation)", type=str)
parser.add_argument("--add_percentage_punctuation", help="Max percentage of white spaces in a sentence to be "
                                                         "prepended with punctuation marks (Punctuation) ", type=int)
parser.add_argument("--remove_percentage_punctuation", help="Max percentage of existing punctuation marks that will be "
                                                            "removed (Punctuation)", type=int)
parser.add_argument("--use_homophones_misspelling", help="If True list of homophones will be used to replace words "
                                                         "(Misspelling)", type=bool)
parser.add_argument("--char_to_remove", help="If specified only that character will be randomly removed. "
                                             "The specified character can also be a white space (RemoveChar)", type=str)
parser.add_argument("--article_swap_probability", help="Probability of applying a transformation. Defaults to 0.5 "
                                                       "(Articles)", type=float)
parser.add_argument("--use_positive_word_masking", help="If True positive (instead of negative) words will be masked "
                                                        "(SentimentMasking)", type=str)

args = parser.parse_args()

# Prepare corruptors
corruptors = []
corruptors_dict = {}


def main():
    if "Articles" in args.corruptors:
        swap_probability = 0.5
        if args.article_swap_probability is not None:
            swap_probability = args.article_swap_probability
        corruptors_dict["Articles"] = Articles(swap_probability)

    if "Digits2Words" in args.corruptors:
        corruptors.append(Digits2Words())
        corruptors_dict["Digits2Words"] = Digits2Words()

    if "Misspellings" in args.corruptors:
        use_homophones = False
        if args.use_homophones_misspelling is not None:
            use_homophones = args.use_homophones_misspelling
        corruptors_dict["Misspelling"] = Misspelling(use_homophones=use_homophones)

    if "Punctuation" in args.corruptors:
        char = ','
        add_percentage = 0
        remove_percentage = 100
        if args.punctuation_char is not None:
            char = args.punctuation_char
        if args.add_percentage_punctuation is not None:
            add_percentage = args.add_percentage_punctuation
        if args.remove_percentage_punctuation is not None:
            remove_percentage = args.remove_percentage_punctuation
        corruptors_dict["Punctuation"] = Punctuation(char=char, add_percentage=add_percentage,
                                                     remove_percentage=remove_percentage)

    if "QWERTY" in args.corruptors:
        if args.words_percentage is not None and args.characters_percentage:
            qwerty = QWERTY(words_percentage=args.words_percentage, characters_percentage=args.characters_percentage)
        elif args.words_percentage is not None:
            qwerty = QWERTY(words_percentage=args.words_percentage)
        elif args.characters_percentage is not None:
            qwerty = QWERTY(characters_percentage=args.characters_percentage)
        else:
            qwerty = QWERTY()
        corruptors_dict["QWERTY"] = qwerty

    if "RemoveChar" in args.corruptors:
        if args.words_percentage is not None and args.characters_percentage:
            remove_char = RemoveChar(char=args.char_to_remove, words_percentage=args.words_percentage, characters_percentage=args.characters_percentage)
        elif args.words_percentage is not None:
            remove_char = RemoveChar(char=args.char_to_remove, words_percentage=args.words_percentage)
        elif args.characters_percentage is not None:
            remove_char = RemoveChar(char=args.char_to_remove, characters_percentage=args.characters_percentage)
        else:
            remove_char = RemoveChar()
        corruptors_dict["RemoveChar"] = remove_char

    if "SentimentMasking" in args.corruptors:
        corruptors_dict["SentimentMasking"] = SentimentMasking()

    if "Swap" in args.corruptors:
        if args.words_percentage is not None:
            swap = Swap(transform_percentage=args.words_percentage)
        else:
            swap = Swap()
        corruptors_dict["Swap"] = swap

    if "ChangeChar" in args.corruptors:
        corruptors_dict["ChangeChar"] = ChangeChar()

    if "WhiteSpaces" in args.corruptors:
        corruptors_dict["WhiteSpaces"] = WhiteSpaces()

    if "AddSubString" in args.corruptors:
        corruptors_dict["AddSubString"] = AddSubString()

    if "SwapWords" in args.corruptors:
        word_creator = None
        if args.word_creator == "synonyms":
            word_creator = SynonymCreator()
        elif args.word_creator == "hypernyms":
            word_creator = HypernymCreator()
        elif args.word_creator == 'word2vec':
            if args.path_to_word2vec_file is None:
                print("you must give --path_to_word2vec_file argument")
                return
            word_creator = Word2VecCreator(args.path_to_word2vec_file)

        selector = None
        if args.selector == "flair_embedding":
            selector = EmbeddingSelector(StackedEmbeddings([FlairEmbeddings('news-forward-fast', use_cache=False),
                                                           FlairEmbeddings('news-backward-fast', use_cache=False)]))
        elif args.selector == "bert_embedding":
            selector = EmbeddingSelector(BertEmbeddings())
        elif args.selector == "elmo_embedding":
            selector = EmbeddingSelector(ELMoEmbeddings())
        elif args.selector == "chrf_teapot":
            selector = TeapotSelector(scorer="chrf")
        elif args.selector == "bleu_teapot":
            selector = TeapotSelector(scorer="bleu")

        if args.words_percentage is not None:
            swap_words = SwapWords(words_percentage=args.words_percentage, selector=selector, word_creator=word_creator)
        else:
            swap_words = SwapWords(selector=selector, word_creator=word_creator)

        corruptors_dict["SwapWords"] = swap_words

    for corruptor in args.corruptors:
        corruptors.append(corruptors_dict[corruptor])

    composed = compose(*corruptors)

    if args.text is not None:
        print(composed(args.text))

    if args.path_to_tsv_file is not None:
        if args.path_to_output_file is None:
            print("you must give --path_to_output_file argument")
        lines = [line.rstrip('\n') for line in open(args.path_to_tsv_file)]
        corrupt_lines = [composed(line) for line in lines]
        with open(args.path_to_output_file, 'w') as f:
            for item in corrupt_lines:
                f.write("%s\n" % item)


main()
