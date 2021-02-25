import pandas as pd 
import json
import numpy as np
from TweetsTextRepresentation import textElaboration

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# hyper-parameters tuning
from sklearn.model_selection import GridSearchCV

# classifiers
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# loading input data
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

# tokenization + stopwords filtering + lemmatization
lemmatized_dataset = textElaboration(dataset)

num_features = 1500
pipes = []
results = []
svm_results = []
nb_results = []
lr_results = []
dt_results = []
cnb_results = []

for num_features in range(1450,1452,1):
	print(num_features)

	# Bag Of Words - uni-grams 
	BOW_uni_svm_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 1))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', svm.SVC()),
			])

	BOW_uni_nb_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 1))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', MultinomialNB()),
			])

	BOW_uni_lr_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 1))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', LogisticRegression()),
			])
	
	BOW_uni_dt_pipe = Pipeline([
		('vect', CountVectorizer(ngram_range=(1, 1))),
		('fselect', SelectKBest(chi2, k=num_features)),
		('clf', DecisionTreeClassifier()),
		])

	BOW_uni_cnb_pipe = Pipeline([
		('vect', CountVectorizer(ngram_range=(1, 1))),
		('fselect', SelectKBest(chi2, k=num_features)),
		('clf', ComplementNB()),
		])

	# TF-IDF - uni-grams
	TFIDF_uni_svm_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 1))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', svm.SVC()),
			])

	TFIDF_uni_nb_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 1))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', MultinomialNB()),
			])

	TFIDF_uni_lr_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 1))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', LogisticRegression()),
			])

	TFIDF_uni_dt_pipe = Pipeline([
		('vect', CountVectorizer(ngram_range=(1, 1))),
		('tfidf', TfidfTransformer()),
		('fselect', SelectKBest(chi2, k=num_features)),
		('clf', DecisionTreeClassifier()),
		])

	TFIDF_uni_cnb_pipe = Pipeline([
		('vect', CountVectorizer(ngram_range=(1, 1))),
		('tfidf', TfidfTransformer()),
		('fselect', SelectKBest(chi2, k=num_features)),
		('clf', ComplementNB()),
		])

	# Bag Of Words - bi-grams 
	BOW_bi_svm_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', svm.SVC()),
			])

	BOW_bi_nb_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', MultinomialNB()),
			])

	BOW_bi_lr_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', LogisticRegression()),
			])

	BOW_bi_dt_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', DecisionTreeClassifier()),
			])

	BOW_bi_cnb_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', ComplementNB()),
			])

	# TF-IDF - bi-grams
	TFIDF_bi_svm_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', svm.SVC()),
			])

	TFIDF_bi_nb_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', MultinomialNB()),
			])

	TFIDF_bi_lr_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', LogisticRegression()),
			])
	
	TFIDF_bi_dt_pipe = Pipeline([
			('vect', CountVectorizer(ngram_range=(1, 2))),
			('tfidf', TfidfTransformer()),
			('fselect', SelectKBest(chi2, k=num_features)),
			('clf', DecisionTreeClassifier()),
			])
	
	TFIDF_bi_cnb_pipe = Pipeline([
		('vect', CountVectorizer(ngram_range=(1, 2))),
		('tfidf', TfidfTransformer()),
		('fselect', SelectKBest(chi2, k=num_features)),
		('clf', ComplementNB()),
		])

	pipes = [BOW_uni_svm_pipe, BOW_bi_svm_pipe, TFIDF_uni_svm_pipe, TFIDF_bi_svm_pipe,
			BOW_uni_nb_pipe, BOW_bi_nb_pipe, TFIDF_uni_nb_pipe, TFIDF_bi_nb_pipe,
			BOW_uni_lr_pipe, BOW_bi_lr_pipe, TFIDF_uni_lr_pipe, TFIDF_bi_lr_pipe,
			BOW_uni_dt_pipe, BOW_bi_dt_pipe, TFIDF_uni_dt_pipe, TFIDF_bi_dt_pipe,
			BOW_uni_cnb_pipe, BOW_bi_cnb_pipe, TFIDF_uni_cnb_pipe, TFIDF_bi_cnb_pipe]

	scoring = {'accuracy' : make_scorer(accuracy_score), 
			'precision' : make_scorer(precision_score,average='micro',labels=labels,zero_division=True),
			'recall' : make_scorer(recall_score,average='micro',labels=labels,zero_division=True), 
			'f1_score' : make_scorer(f1_score,average='micro',labels=labels,zero_division=True)}

	# store results 
	for pipe in pipes:
		results.append(cross_validate(estimator= pipe,
						X=lemmatized_dataset,
						y=labels,
						cv=10,
						scoring=scoring
						))      

	# append svm results 
	tmp_svm = {}
	tmp_svm["Num Features"] = num_features
	tmp_svm["BOW uni-grams Accuracy"] = np.mean(results[0]['test_accuracy'])
	tmp_svm["BOW uni-grams Precision"] = np.mean(results[0]['test_precision'])
	tmp_svm["BOW uni-grams Recall"] = np.mean(results[0]['test_recall'])
	tmp_svm["BOW uni-grams F1 Score"] = np.mean(results[0]['test_f1_score'])

	tmp_svm["BOW bi-grams Accuracy"] = np.mean(results[1]['test_accuracy'])
	tmp_svm["BOW bi-grams Precision"] = np.mean(results[1]['test_precision'])
	tmp_svm["BOW bi-grams Recall"] = np.mean(results[1]['test_recall'])
	tmp_svm["BOW bi-grams F1 Score"] = np.mean(results[1]['test_f1_score'])

	tmp_svm["TFIDF uni-grams Accuracy"] = np.mean(results[2]['test_accuracy'])
	tmp_svm["TFIDF uni-grams Precision"] = np.mean(results[2]['test_precision'])
	tmp_svm["TFIDF uni-grams Recall"] = np.mean(results[2]['test_recall'])
	tmp_svm["TFIDF uni-grams F1 Score"] = np.mean(results[2]['test_f1_score'])

	tmp_svm["TFIDF bi-grams Accuracy"] = np.mean(results[3]['test_accuracy'])
	tmp_svm["TFIDF bi-grams Precision"] = np.mean(results[3]['test_precision'])
	tmp_svm["TFIDF bi-grams Recall"] = np.mean(results[3]['test_recall'])
	tmp_svm["TFIDF bi-grams F1 Score"] = np.mean(results[3]['test_f1_score'])
	svm_results.append(tmp_svm)
	#print(svm_results)

	# append NB results 
	tmp_nb = {}
	tmp_nb["Num Features"] = num_features
	tmp_nb["BOW uni-grams Accuracy"] = np.mean(results[4]['test_accuracy'])
	tmp_nb["BOW uni-grams Precision"] = np.mean(results[4]['test_precision'])
	tmp_nb["BOW uni-grams Recall"] = np.mean(results[4]['test_recall'])
	tmp_nb["BOW uni-grams F1 Score"] = np.mean(results[4]['test_f1_score'])

	tmp_nb["BOW bi-grams Accuracy"] = np.mean(results[5]['test_accuracy'])
	tmp_nb["BOW bi-grams Precision"] = np.mean(results[5]['test_precision'])
	tmp_nb["BOW bi-grams Recall"] = np.mean(results[5]['test_recall'])
	tmp_nb["BOW bi-grams F1 Score"] = np.mean(results[5]['test_f1_score'])

	tmp_nb["TFIDF uni-grams Accuracy"] = np.mean(results[6]['test_accuracy'])
	tmp_nb["TFIDF uni-grams Precision"] = np.mean(results[6]['test_precision'])
	tmp_nb["TFIDF uni-grams Recall"] = np.mean(results[6]['test_recall'])
	tmp_nb["TFIDF uni-grams F1 Score"] = np.mean(results[6]['test_f1_score'])

	tmp_nb["TFIDF bi-grams Accuracy"] = np.mean(results[7]['test_accuracy'])
	tmp_nb["TFIDF bi-grams Precision"] = np.mean(results[7]['test_precision'])
	tmp_nb["TFIDF bi-grams Recall"] = np.mean(results[7]['test_recall'])
	tmp_nb["TFIDF bi-grams F1 Score"] = np.mean(results[7]['test_f1_score'])
	nb_results.append(tmp_nb)
	#print(nb_results)

	# append LR results 
	tmp_lr = {}
	tmp_lr["Num Features"] = num_features
	tmp_lr["BOW uni-grams Accuracy"] = np.mean(results[8]['test_accuracy'])
	tmp_lr["BOW uni-grams Precision"] = np.mean(results[8]['test_precision'])
	tmp_lr["BOW uni-grams Recall"] = np.mean(results[8]['test_recall'])
	tmp_lr["BOW uni-grams F1 Score"] = np.mean(results[8]['test_f1_score'])

	tmp_lr["BOW bi-grams Accuracy"] = np.mean(results[9]['test_accuracy'])
	tmp_lr["BOW bi-grams Precision"] = np.mean(results[9]['test_precision'])
	tmp_lr["BOW bi-grams Recall"] = np.mean(results[9]['test_recall'])
	tmp_lr["BOW bi-grams F1 Score"] = np.mean(results[9]['test_f1_score'])

	tmp_lr["TFIDF uni-grams Accuracy"] = np.mean(results[10]['test_accuracy'])
	tmp_lr["TFIDF uni-grams Precision"] = np.mean(results[10]['test_precision'])
	tmp_lr["TFIDF uni-grams Recall"] = np.mean(results[10]['test_recall'])
	tmp_lr["TFIDF uni-grams F1 Score"] = np.mean(results[10]['test_f1_score'])

	tmp_lr["TFIDF bi-grams Accuracy"] = np.mean(results[11]['test_accuracy'])
	tmp_lr["TFIDF bi-grams Precision"] = np.mean(results[11]['test_precision'])
	tmp_lr["TFIDF bi-grams Recall"] = np.mean(results[11]['test_recall'])
	tmp_lr["TFIDF bi-grams F1 Score"] = np.mean(results[11]['test_f1_score'])
	lr_results.append(tmp_lr)
	#print(lr_results)

	# append DT results 
	tmp_dt = {}
	tmp_dt["Num Features"] = num_features
	tmp_dt["BOW uni-grams Accuracy"] = np.mean(results[12]['test_accuracy'])
	tmp_dt["BOW uni-grams Precision"] = np.mean(results[12]['test_precision'])
	tmp_dt["BOW uni-grams Recall"] = np.mean(results[12]['test_recall'])
	tmp_dt["BOW uni-grams F1 Score"] = np.mean(results[12]['test_f1_score'])

	tmp_dt["BOW bi-grams Accuracy"] = np.mean(results[13]['test_accuracy'])
	tmp_dt["BOW bi-grams Precision"] = np.mean(results[13]['test_precision'])
	tmp_dt["BOW bi-grams Recall"] = np.mean(results[13]['test_recall'])
	tmp_dt["BOW bi-grams F1 Score"] = np.mean(results[13]['test_f1_score'])

	tmp_dt["TFIDF uni-grams Accuracy"] = np.mean(results[14]['test_accuracy'])
	tmp_dt["TFIDF uni-grams Precision"] = np.mean(results[14]['test_precision'])
	tmp_dt["TFIDF uni-grams Recall"] = np.mean(results[14]['test_recall'])
	tmp_dt["TFIDF uni-grams F1 Score"] = np.mean(results[14]['test_f1_score'])

	tmp_dt["TFIDF bi-grams Accuracy"] = np.mean(results[15]['test_accuracy'])
	tmp_dt["TFIDF bi-grams Precision"] = np.mean(results[15]['test_precision'])
	tmp_dt["TFIDF bi-grams Recall"] = np.mean(results[15]['test_recall'])
	tmp_dt["TFIDF bi-grams F1 Score"] = np.mean(results[15]['test_f1_score'])
	dt_results.append(tmp_dt)
	#print(dt_results)

	# append CNB results 
	tmp_cnb = {}
	tmp_cnb["Num Features"] = num_features
	tmp_cnb["BOW uni-grams Accuracy"] = np.mean(results[16]['test_accuracy'])
	tmp_cnb["BOW uni-grams Precision"] = np.mean(results[16]['test_precision'])
	tmp_cnb["BOW uni-grams Recall"] = np.mean(results[16]['test_recall'])
	tmp_cnb["BOW uni-grams F1 Score"] = np.mean(results[16]['test_f1_score'])

	tmp_cnb["BOW bi-grams Accuracy"] = np.mean(results[17]['test_accuracy'])
	tmp_cnb["BOW bi-grams Precision"] = np.mean(results[17]['test_precision'])
	tmp_cnb["BOW bi-grams Recall"] = np.mean(results[17]['test_recall'])
	tmp_cnb["BOW bi-grams F1 Score"] = np.mean(results[17]['test_f1_score'])

	tmp_cnb["TFIDF uni-grams Accuracy"] = np.mean(results[18]['test_accuracy'])
	tmp_cnb["TFIDF uni-grams Precision"] = np.mean(results[18]['test_precision'])
	tmp_cnb["TFIDF uni-grams Recall"] = np.mean(results[18]['test_recall'])
	tmp_cnb["TFIDF uni-grams F1 Score"] = np.mean(results[18]['test_f1_score'])

	tmp_cnb["TFIDF bi-grams Accuracy"] = np.mean(results[19]['test_accuracy'])
	tmp_cnb["TFIDF bi-grams Precision"] = np.mean(results[19]['test_precision'])
	tmp_cnb["TFIDF bi-grams Recall"] = np.mean(results[19]['test_recall'])
	tmp_cnb["TFIDF bi-grams F1 Score"] = np.mean(results[19]['test_f1_score'])
	cnb_results.append(tmp_cnb)
	#print(dt_results)

# save results on files
with open('./FS_Results/FS_svm_results', 'w') as fout:
    json.dump(svm_results, fout, indent=4)

with open('./FS_Results/FS_nb_results', 'w') as fout:
    json.dump(nb_results, fout, indent=4)

with open('./FS_Results/FS_lr_results', 'w') as fout:
    json.dump(lr_results, fout, indent=4)

with open('./FS_Results/FS_dt_results', 'w') as fout:
    json.dump(dt_results, fout, indent=4)

with open('./FS_Results/FS_cnb_results', 'w') as fout:
    json.dump(cnb_results, fout, indent=4)