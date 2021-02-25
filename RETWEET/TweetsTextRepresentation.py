import pandas as pd 
import nltk 
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist, classify, NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# stopwords filtering function 
def removeStopWords(stopWords, tokens):
    cleaned_tokens = []
    cleaned_tokens = [word for word in tokens if word not in stopWords]
    return cleaned_tokens

# lemmatization function
def lemmatize(tokens):
    wn.ensure_loaded()
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    lemmatized_sentence_joined = ""
    
    # pos tagging & lemmatization
    for word, tag in pos_tag(tokens):

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    
    # joining and removing stopwords
    for lemma in lemmatized_sentence:
        lemmatized_sentence_joined += lemma + " "

    return lemmatized_sentence_joined

# splitting a list of tweets into a list of words 
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# loading input data
dataset = pd.read_csv("./Dataset/merged_weeks_1_2.csv")
dataset.dropna(subset=["tag"], inplace=True)

# tokenizaion + stopwords filtering + lemmatization function
# returns a list where each element is a lemmatized tweet 
def textElaboration(dataset, key_name):
    stop_words = stopwords.words('english')
    lemmatized_set = []

    for row_index in dataset.index:
        row_field = dataset.loc[row_index, key_name]

        # tokenization
        tokens = nltk.word_tokenize(row_field)

        # stopwords filtering
        filtered_tokens = removeStopWords(stop_words, tokens)

        # lemmatization
        lemmatized_set.append(lemmatize(filtered_tokens).strip())
    
    return lemmatized_set


