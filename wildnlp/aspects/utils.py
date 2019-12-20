import functools
from flair.data import Sentence
import numpy as np
import random
import copy


def replace_token(sentence: Sentence, token, new_word=""):
    """Replaces token with word."""
    if new_word == "":
        sentence.tokens.remove(token)
    else:
        token.text = new_word


def compose(*functions):
    """Chains multiple aspects into a single function.

    :param functions: Object(s) of the Callable instance.

    :return: chained function

    Example::

        from wildnlp.aspects.utils import compose
        from wildnlp.aspects import Swap, QWERTY

        composed_aspect = compose(Swap(), QWERTY())
        modified_text = composed_aspect('Text to corrupt')

    """
    return functools.reduce(lambda f, g: lambda x: g(f(x)),
                            functions, lambda x: x)


def word_dropout(text, if_percent=False, how_many=1, seed=42):
    """Returns text with deleted random words.
    :param text: Text to corrupt
    :param if_percent: if true parameter how_many means how many percent words in text function should corrupt
    :param how_many: number of changes in text or percent of changes in text when if_percent = True (default: 1)
    :param seed: Random seed.

    :return modified text
    """

    random.seed(seed)

    sentence = Sentence(text, True)
    tokens = sentence.tokens
    tokens_len = len(tokens)

    if if_percent:
        if how_many < 1:
            how_many = int(how_many*len(tokens))
        else:
            how_many = int(how_many*len(tokens)/100)

    if how_many > tokens_len:
        how_many = tokens_len

    tokens_to_remove = []
    for idx in random.sample(range(tokens_len), tokens_len):
        if how_many == 0:
            break

        tokens_to_remove.append(tokens[idx])
        how_many -= 1

    for token in tokens_to_remove:
        tokens.remove(token)

    return sentence.to_plain_string()


def change_random_word(text, corruptor, if_percent=False, how_many=1, seed=42) -> str:
    """Returns text with corrupt random words.
    :param text: Text to corrupt
    :param corruptor: chained function from compose
    :param if_percent: if true parameter how_many means how many percent words in text function should corrupt
    :param how_many: number of changes in text or percent of changes in text when if_percent = True (default: 1)
    :param seed: Random seed.

    :return corrupted text
    """
    random.seed(seed)

    sentence = Sentence(text, True)
    tokens = sentence.tokens
    tokens_len = len(tokens)

    if if_percent:
        if how_many < 1:
            how_many = int(how_many*len(tokens))
        else:
            how_many = int(how_many*len(tokens)/100)

        if how_many < 1:
            how_many = 1

    if how_many > tokens_len:
        how_many = tokens_len

    for idx in random.sample(range(tokens_len), tokens_len):
        if how_many == 0:
            break

        new_word = corruptor(tokens[idx].text)
        if new_word != tokens[idx].text:
            how_many -= 1

        tokens[idx].text = new_word

    return sentence.to_plain_string()


def change_most_important_word(text, label, pipeline, corruptor, if_percent=False, how_many=1) -> str:
    """Returns text with corrupt most important word for classification task.
    :param text: Text to corrupt
    :param label: Label of text to corrupt
    :param pipeline: Function that takes list of str (inputs) as argument and returns np.array
        (probabilities of classes) e.g. predict_proba from sklearn
    :param corruptor: chained function from compose
    :param if_percent: if true parameter how_many means how many percent words in text function should corrupt
    :param how_many: number of changes in text or percent of changes in text when if_percent = True (default: 1)

    :return corrupted text
    """
    sentence = Sentence(text, True)
    tokens = sentence.tokens
    if len(tokens) == 1:
        tokens[0].text = corruptor(tokens[0].text)
        return sentence.to_plain_string()

    sorted_tokens = find_most_important_words(tokens, label, pipeline)

    if if_percent:
        if how_many < 1:
            how_many = int(how_many*len(tokens))
        else:
            how_many = int(how_many*len(tokens)/100)

    if how_many > len(sorted_tokens):
        how_many = len(sorted_tokens)

    for token in sorted_tokens:
        if how_many == 0:
            break

        new_word = corruptor(token.text)
        if new_word != token.text:
            how_many -= 1

        token.text = new_word

    return sentence.to_plain_string()


def change_model_prediction(text, label, pipeline, corruptors, verbose=False):
    """Returns text with corrupt words if model gave correct prediction in classification task
    for original text or original text otherwise and True if model gave incorrect prediction for corrupt text or False
    if model gave correct prediction for corrupt text or incorrect prediction for original text.
    :param text: Text to corrupt
    :param label: Label of text to corrupt
    :param pipeline: Function that takes list of str (inputs) as argument and returns np.array
        (probabilities of classes) e.g. predict_proba from sklearn
    :param corruptors: list of chained functions from compose
    :param verbose: if True function print corrupt texts and their scores (default: False)

    :return corrupted text, True or False
    """
    sentence = Sentence(text, True)
    tokens = sentence.tokens

    sorted_tokens = find_most_important_words(tokens, label, pipeline)

    current_text = text
    for token in sorted_tokens:
        new_text, bug_text, flip = select_bug(current_text, tokens, token, label, pipeline, corruptors, verbose)
        # corruption of word increase model certainty
        if new_text == "":
            continue

        current_text = new_text
        token.text = bug_text
        # wrong model prediction
        if flip:
            return current_text, True

    # correct model prediction
    return current_text, False


def select_bug(text, tokens, token_to_corrupt,  label, pipeline, corruptors, verbose=False):
    """For sentence and word returns new sentence with this word changed in a way that causes the largest loss
    increase.
    :param text: text to corrupt
    :param tokens: tokens from text to corrupt
    :param token_to_corrupt: token to corrupt
    :param label: Label of text to corrupt
    :param pipeline: Function that takes list of str (inputs) as argument and returns np.array
        (probabilities of classes) e.g. predict_proba from sklearn
    :param corruptors: list of chained functions from compose
    :param verbose: if True function print corrupt texts and their scores (default: False)

    :return corrupted text
    """
    bugs = [corrupt(token_to_corrupt.text) for corrupt in corruptors]
    new_texts = [None] * len(bugs)

    # prepare list of sentences with bugged word
    for i, bug in enumerate(bugs):
        sentence_tmp = Sentence()
        for t in tokens:
            sentence_tmp.add_token(copy.deepcopy(t))
        word_tmp = sentence_tmp.tokens[token_to_corrupt.idx - 1]
        replace_token(sentence_tmp, word_tmp, bug)
        new_texts[i] = sentence_tmp.to_plain_string()

    # classification on modified sentences and original sentence
    outputs = pipeline(new_texts + [text])
    original_output = outputs[len(outputs) - 1]

    max_score = 0
    best_text = ""
    best_bug = ""
    flip = False

    if verbose:
        print("------------------------------------------------------")
        print("Input text:", text)
        print("Word to corrupt: ", token_to_corrupt.text)

    # find sentence with the largest loss increase
    for i, bug in enumerate(bugs):
        score = original_output[label] - outputs[i][label]
        if verbose:
            print("Corrupt text: ", new_texts[i])
            print("Score: ", outputs[i][label])
        if score > max_score:
            max_score = score
            best_text = new_texts[i]
            best_bug = bug
            new_label = np.argmax(outputs[i])
            if new_label != label:
                flip = True

    return best_text, best_bug, flip


def find_most_important_words(tokens, label, pipeline):
    """Returns tokens sorted by importance for classification task.
    :param tokens: Tokens to sort
    :param label: Label of tokens
    :param pipeline: Function that takes list of str (inputs) as argument and returns np.array
        (probabilities of classes) e.g. predict_proba from sklearn

    :return sorted tokens
    """
    if len(tokens) == 1:
        return tokens

    true_probability = [None] * len(tokens)
    new_texts = [""] * len(tokens)

    # create list of modified sentences with removed words
    for i, token in enumerate(tokens):
        sentence_tmp = Sentence()
        for t in tokens:
            sentence_tmp.add_token(t)
        replace_token(sentence_tmp, token)
        new_texts[i] = sentence_tmp.to_plain_string()

    output = pipeline(new_texts)

    # probability of of true label with removed word
    for i, o in enumerate(output):
        true_probability[i] = o[label]

    sorted_tokens = [token for p, token in sorted(zip(true_probability, tokens), key=lambda pair: pair[0])]

    return sorted_tokens
