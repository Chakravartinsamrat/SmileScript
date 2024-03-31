import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
#initializes stemmer, stemmer is used to reduce words down to thier root form
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
# sentence to indivual words
def stem(word):
    return stemmer.stem(word.lower())
#defines func stem input words (converts to lower before stemmming)

def bag_of_words(tokenize_sentence,all_words):
    #a func that takes a tokenized sentence and list of all words as input
    tokenize_sentence=[stem(w) for w in tokenize_sentence]

    bag=np.zeros(len(all_words), dtype=np.float32)
    #initializes a numpy array of zeros with the same lenght as the the list of all words
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx]=1.0
    #if the word is present in the tokenizes and stemmed sentence it sets the corresponding index int he bag array to 1.0
    return bag

