import pandas as pd
from TweetsTextRepresentation import textElaboration
import joblib
import numpy as np

from nltk.probability import FreqDist

# ------------------------- loading input data ------------------------- #
dataset = pd.read_csv("./Dataset/merged_weeks_1_2_3_4.csv")
dataset.dropna(subset=["tag"], inplace=True)

# ------------------------- tokenization + stopwords filtering + lemmatization ------------------------- #
lemmatized_dataset = textElaboration(dataset, "Tweet Text")

#get vocabulary
joined_dataset = " ".join(lemmatized_dataset)
print((len(set(joined_dataset.split(" ")))))

"""
# ------------------------- saving lemmatized dataset on disk ------------------------- #
with open("./Dataset/Lemmatized_Dataset.txt", "a", encoding="utf-8") as text_file:
    for tweet in lemmatized_dataset:
        text_file.write(tweet)
        text_file.write("\n")
"""
