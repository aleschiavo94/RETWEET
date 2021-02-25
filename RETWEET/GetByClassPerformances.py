import pandas as pd 
import json
import time
import math
import numpy as np
import joblib
from TweetsTextRepresentation import textElaboration

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

# classifiers
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import paired_ttest_5x2cv

# ensemble 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


# ------------------------- loading input data ------------------------- #
dataset = pd.read_csv("./Dataset/merged_weeks_1_2_3_4.csv")
dataset.dropna(subset=["tag"], inplace=True)

# label coding 
tag_codes = {
    "positive" : 1, 
    "negative" : 0,
    "neutral" : -1
}

# category mapping
dataset["tag_code"] = dataset["tag"]
dataset = dataset.replace({"tag_code" :tag_codes})

# labels set    
labels = dataset["tag_code"]

# ------------------------- tokenization + stopwords filtering + lemmatization ------------------------- #
lemmatized_dataset = textElaboration(dataset, "Tweet Text")

# 5) SVM TF-IDF - uni-grams
TFIDF_uni_svm_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.75)),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2, k=3000)),
    ('clf', svm.SVC(C=10, gamma=0.1)),
    ])

# 4) Logistic Regression BOW - uni-grams
BOW_uni_lr_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.75)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', LogisticRegression(C=1, max_iter=1500)),
    ])

# 3) RANDOM FOREST TF-IDF - uni-grams
TFIDF_uni_rf_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.65)),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2, k=2000)),
  ('clf', RandomForestClassifier(criterion="gini", min_samples_split=10, n_estimators=1200)),
  ])

# 2) Bagging + SVM TF-IDF - uni-grams 
TFIDF_uni_bg_svm_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.85)),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2, k=4000)),
  ('clf', BaggingClassifier(base_estimator=svm.SVC(), n_estimators=30)),
  ])

X, y = shuffle(lemmatized_dataset, labels, random_state=3*320391)

predictions = cross_val_predict(estimator=TFIDF_uni_bg_svm_pipe,
            X=X,
            y=y,
            cv=10,
            n_jobs=-1
            )      

print(classification_report(y, predictions, target_names=["Neutral", "Negative", "Positive"], digits=4))