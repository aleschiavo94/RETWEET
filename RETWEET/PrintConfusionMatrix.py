import pandas as pd
from TweetsTextRepresentation import textElaboration
import joblib
import numpy as np
import matplotlib.pyplot as plt

from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


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
class_names = ["Neutral", "Negative", "Positive"]

# ------------------------- tokenization + stopwords filtering + lemmatization ------------------------- #
lemmatized_dataset = textElaboration(dataset, "Tweet Text")

loaded_model = joblib.load('./Tuned_Models/SVM_2.pkl')

np.set_printoptions(precision=2)

# ------------------------- Plot non-normalized confusion matrix ------------------------- #
title = "Confusion Matrix"
disp = plot_confusion_matrix(loaded_model, lemmatized_dataset, labels,
                                display_labels=class_names,
                                cmap="gist_heat",
                                normalize=None)
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()
