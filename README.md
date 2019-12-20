[![Documentation Status](https://readthedocs.org/projects/nlp-demo/badge/?version=latest)](https://nlp-demo.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/wild-nlp)](https://pepy.tech/project/wild-nlp)
[![PyPI version](https://badge.fury.io/py/wild-nlp.svg)](https://badge.fury.io/py/wild-nlp)

![alt wildnlp-logo](logo.png)  

Corrupt an input text to test NLP models' robustness.  
For details refer to https://nlp-demo.readthedocs.io

## Installation
`python setup.py install`

## Supported aspects
All together we defined and implemented 11 aspects of text corruption.

1. **Articles**
   
   Randomly removes or swaps articles into wrong ones.

2. **Digits2Words**

   Converts numbers into words. Handles floating numbers as well.

3. **Misspellings**

   Misspells words appearing in the Wikipedia list of:  
    * **commonly misspelled English words**  
    * **homophones**

4. **Punctuation**

   Randomly adds or removes specified punctuation marks.

5. **QWERTY**

   Simulates errors made while writing on a QWERTY-type keyboard.

6. **RemoveChar**

   Randomly removes:  
   * **characters** from words or  
   * **white spaces** from sentences

7. **SentimentMasking**

   Replaces random, single character with for example an asterisk in:  
   * **negative** or  
   * **positive** words from Opinion Lexicon:    
   http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

8. **Swap**

   Randomly swaps two characters within a word, excluding punctuations.
   
9. **Change char**

   Randomly change characters according to chosen dictionary, default is 'ocr' to simulate simple OCR errors.
   
10. **White spaces**

   Randomly add or remove white spaces (listed as a parameter).

11. **Sub string**

   Randomly add a substring to simulate more comples signs.
   
12. **Swap words**

   Swap words for synonyms or hypernyms. Function takes as argument object word_creator which for word returns list of words which can replace it 
   and object selector which for word and list of words which can replace it returns word choosen from list.<br/>
   Function can take custom word\_creator and selector, but some word\_creators and selectors are already implemented.<br/>
   word_creator:
   * SynonymCreator: creates syonyms from WordNet
   * HypernymCreator: creates hypernym from WordNet
   * Word2VecCreator: find most similar words in Word2Vec model (e.g. Glove) 

   selector:
   * random selector
   * EmbeddingSelector: takes as argument embedding implemented in flair (e. g. FlairEmbeddings, BertEmbeddings, ELMoEmbeddings)
   * TeapotSelector: select word based on similarity measured by ChrF, METEOR, BLEU
   
   
```diff
- All aspects can be chained together with the wildnlp.aspects.utils.compose function.
```

## Additional functions
1. **change_most_important**

    Change some number of most important words in text using corruptor given as argument. Function works only on classification task. 
    Importance of words is measure by how much confidence of choosing correct class decrease without that word. 

2. **change_model_prediction**

    Change most important words in text using corruptors given as argument until the result of the classification changes.  
    Importance of words is measure by how much confidence of choosing correct class decrease without that word. 
    Corruptor deacreasing confidence most is chosen for every word.

3. **change_random_word**

    Change some number of random of words in text using corruptor given as argument.

## Supported datasets
Aspects can be applied to any text. Below is the list of datasets for which we already implemented processing pipelines. 

1. **CoNLL**

   The CoNLL-2003 shared task data for language-independent named entity recognition.

2. **IMDB**

   The IMDB dataset containing movie reviews for a sentiment analysis. The dataset consists of 50 000 reviews of two classes, negative and positive.

3. **SNLI**

   The SNLI dataset supporting the task of natural language inference.

4. **SQuAD**

   The SQuAD dataset for the Machine Comprehension problem.

## Usage
```python
from wildnlp.aspects.dummy import Reverser, PigLatin
from wildnlp.aspects.utils import compose
from wildnlp.datasets import SampleDataset

# Create a dataset object and load the dataset
dataset = SampleDataset()
dataset.load()

# Crate a composed corruptor function.
# Functions will be applied in the same order they appear.
composed = compose(Reverser(), PigLatin())

# Apply the function to the dataset
modified = dataset.apply(composed)
```

## Console interface
User can use WildNLP using console via wildnlp.py file. For example:

```shell
python wildnlp.py --text "Some example test to corrupt" --corruptors Swap QWERTY
python wildnlp.py --path_to_tsv_file file.tsv --path_to_output_file output_file.tsv --corruptors Swap QWERTY
```
Options:
* **--corruptors:** corrupt functions; type: list of str; choices: Articles, Digits2Words, Misspellings, Punctuation, QWERTY, RemoveChar, SentimentMasking,
                     Swap, ChangeChar, WhiteSpaces, AddSubString, SwapWords
* **--text:** text to corrupt; type: str
* **--path\_to\_tsv\_file:** path to tsv file with texts to corrupt; type: str
* **--path\_to\_output\_file:** path to output file; type: str
* **--words_percentage:** percentage of words to corrupt (for functions: RemoveChar, Swap, QWERTY, SwapWords); type: int
* **--characters_percentage:** percentage of characters to corrupt (for functions: RemoveChar, QWERTY); type: str
* **--word_creator:** word creator for SwapWords; type: str; choices: synonyms, hypernyms, word2vec; default: hypernyms
* **--path\_to\_word2vec\_file:** path to embeddings in word2vec format file; type: str
* **--selector:** selector for SwapWord; type: str; choices: random, flair\_embedding, bert\_embedding, elmo\_embedding, chrf\_teapot, bleu\_teapot; default: random
* **--pos:** part of speech to change by SwapWords (v - verb, r - adverb, a - adjective, n - noun ); type: list of str; choices: v, r, a, n

