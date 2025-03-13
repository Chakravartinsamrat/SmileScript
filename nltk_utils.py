import nltk
import numpy as np
#nltk.download('punkt')
#nltk.donwload('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
# sentence to indivual words
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatize(w) for w in tokenized_sentence]

    bag=np.zeros(len(all_words), dtype=np.float32)
    #initializes a numpy array of zeros with the same lenght as the the list of all words
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0

    #if the word is present in the tokenizes and stemmed sentence it sets the corresponding index int he bag array to 1.0
    return bag

    