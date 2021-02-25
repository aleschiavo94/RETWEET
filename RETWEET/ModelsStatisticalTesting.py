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
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle

# classifiers
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import paired_ttest_5x2cv

# ensemble 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# function that computes paired t-statistic to determine
# whether the two models are significantly different
def paired_tTest(pipeline_A, pipeline_B, dataset, labels, iter):
    results_A = []
    results_B = []
    model_A = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }
    model_B = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }

    # 30 times 10-fold-cross-cvalidation 
    for i in range(1,iter+1):
        X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
        if (i%10 == 0):
            print("Iteration # {}..." .format(i))

        results_A.append(cross_validate(estimator=pipeline_A,
                    X=X,
                    y=y,
                    cv=10,
                    scoring=scoring,
                    n_jobs=-1
                    ))      
        results_B.append(cross_validate(estimator=pipeline_B,
                    X=X,
                    y=y,
                    cv=10,
                    scoring=scoring,
                    n_jobs=-1
                    ))      

    for result in results_A:
        model_A["Accuracy"].append(np.mean(result['test_accuracy']))
        model_A["Precision"].append(np.mean(result['test_precision']))
        model_A["Recall"].append(np.mean(result['test_recall']))
        model_A["F1 Score"].append(np.mean(result['test_f1_score']))

    for result in results_B:
        model_B["Accuracy"].append(np.mean(result['test_accuracy']))
        model_B["Precision"].append(np.mean(result['test_precision']))
        model_B["Recall"].append(np.mean(result['test_recall']))
        model_B["F1 Score"].append(np.mean(result['test_f1_score']))

    mean_accuracy_A = np.mean(model_A["Accuracy"])
    mean_accuracy_B = np.mean(model_B["Accuracy"])
    delta_means = mean_accuracy_A - mean_accuracy_B
    delta_sum = 0

    print("Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)" % (np.mean(model_A["Accuracy"]), np.std(model_A["Accuracy"]) * 2))
    print("Mean Accuracy rate of model M2: %0.4f (+/- %0.4f)" % (np.mean(model_B["Accuracy"]), np.std(model_B["Accuracy"]) * 2))

    # compute the Variance of the difference between the two models
    for i in range(0,iter):
        delta_i = model_A["Accuracy"][i] - model_B["Accuracy"][i]
        delta_sum += (delta_i - delta_means)*(delta_i - delta_means)
    variance = delta_sum/iter

    # compute t-statistic 
    t_statistic = delta_means/math.sqrt(variance/iter)
    print("t-statistic value is: {}" .format(round(t_statistic,5)))

    # t-table look-up
    t_table = pd.read_csv("./Utils/t-table.csv") 
    if(t_statistic > t_table.loc[4, "0.025"] or t_statistic < - t_table.loc[4, "0.025"]):
        print("t-table value is: {}" .format(t_table.loc[4, "0.025"]))
        print("The Null Hypothesis can be rejected. The two classifiers are statistically significant different.")
    else:
        print("t-table value is: {}" .format(t_table.loc[4, "0.025"]))
        print("The t-statistic value does not lie in the rejection region.")
    return t_statistic


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

# scoring function
scoring = {'accuracy' : make_scorer(accuracy_score), 
        'precision' : make_scorer(precision_score,average='micro',labels=labels,zero_division=True),
        'recall' : make_scorer(recall_score,average='micro',labels=labels,zero_division=True), 
        'f1_score' : make_scorer(f1_score,average='micro',labels=labels,zero_division=True)}

# 1) Bagging + DT TF-IDF - uni-grams 
TFIDF_uni_bg_dt_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.65)),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2, k=1000)),
  ('clf', BaggingClassifier(n_estimators=100)),
  ])

# 2) Bagging + SVM TF-IDF - uni-grams 
TFIDF_uni_bg_svm_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.85)),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2, k=4000)),
  ('clf', BaggingClassifier(base_estimator=svm.SVC(), n_estimators=30)),
  ])

# 3) RANDOM FOREST TF-IDF - uni-grams
TFIDF_uni_rf_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.65)),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2, k=2000)),
  ('clf', RandomForestClassifier(criterion="gini", min_samples_split=10, n_estimators=1200)),
  ])

# 4) Logistic Regression BOW - uni-grams
BOW_uni_lr_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.75)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', LogisticRegression(C=1, max_iter=1500)),
    ])

# 5) SVM TF-IDF - uni-grams
TFIDF_uni_svm_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.75)),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2, k=3000)),
    ('clf', svm.SVC(C=10, gamma=0.1)),
    ])

models_pipes = [{"model name": "Bagging + DT uni-grams TF-IDF", "model": TFIDF_uni_bg_dt_pipe}, 
                {"model name": "Bagging + SVM uni-grams TF-IDF", "model": TFIDF_uni_bg_svm_pipe},
                {"model name": "Random Forest uni-grams TF-IDF", "model": TFIDF_uni_rf_pipe},
                {"model name": "Logistic Regression uni-grams BOW", "model": BOW_uni_lr_pipe},
                {"model name": "SVM uni-grams TF-IDF", "model": TFIDF_uni_svm_pipe}]

numOfRuns = 10

# cross testing between best models (paired t-test)
for model_1 in models_pipes:
    for model_2 in models_pipes:
        if model_1 == model_2:
            continue
        print("-----------------------------------------------------------------------------------------")
        print("Testing Models:   M1: {}    M2: {}".format(model_1["model name"], model_2["model name"]))
        paired_tTest(model_1["model"], model_2["model"], lemmatized_dataset, labels, numOfRuns)




