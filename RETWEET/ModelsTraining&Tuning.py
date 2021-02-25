import pandas as pd 
import json
import time
import numpy as np
from TweetsTextRepresentation import textElaboration

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
import joblib

# hyper-parameters tuning
from sklearn.model_selection import GridSearchCV

# base classifiers
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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
VOCABULARY_SIZE = 4688


# scoring function
scoring = {'accuracy' : make_scorer(accuracy_score), 
			'precision' : make_scorer(precision_score,average='micro',labels=labels,zero_division=True),
			'recall' : make_scorer(recall_score,average='micro',labels=labels,zero_division=True), 
			'f1_score' : make_scorer(f1_score,average='micro',labels=labels,zero_division=True)}

# ------------------------- defining pipelines ------------------------- #

# Bag Of Words - uni-grams 
BOW_uni_svm_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
    ])

BOW_uni_nb_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('fselect', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
    ])

BOW_uni_lr_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('fselect', SelectKBest(chi2)),
    ('clf', LogisticRegression()),
    ])

BOW_uni_cnb_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('fselect', SelectKBest(chi2)),
  ('clf', ComplementNB()),
  ])

BOW_uni_dt_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('fselect', SelectKBest(chi2)),
  ('clf', DecisionTreeClassifier()),
  ])

BOW_uni_bg_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('fselect', SelectKBest(chi2)),
  ('clf', BaggingClassifier()),
  ])

BOW_uni_rf_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('fselect', SelectKBest(chi2)),
  ('clf', RandomForestClassifier())
  ])

# TF-IDF - uni-grams
TFIDF_uni_svm_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
    ])

TFIDF_uni_nb_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
    ])

TFIDF_uni_lr_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', LogisticRegression()),
    ])

TFIDF_uni_cnb_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2,)),
  ('clf', ComplementNB()),
  ])

TFIDF_uni_dt_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2,)),
  ('clf', DecisionTreeClassifier()),
  ])

TFIDF_uni_bg_dt_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2,)),
  ('clf', BaggingClassifier()),
  ])

TFIDF_uni_bg_svm_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2,)),
  ('clf', BaggingClassifier(base_estimator=svm.SVC())),
  ])

TFIDF_uni_bg_lr_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2,)),
  ('clf', BaggingClassifier(base_estimator=LogisticRegression())),
  ])

TFIDF_uni_rf_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 1))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2,)),
  ('clf', RandomForestClassifier()),
  ])

# Bag Of Words - bi-grams 
BOW_bi_svm_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
    ])

BOW_bi_nb_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
    ])

BOW_bi_lr_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', LogisticRegression()),
    ])

BOW_bi_cnb_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', ComplementNB()),
    ])

BOW_bi_dt_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', DecisionTreeClassifier()),
    ])

BOW_bi_bg_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', BaggingClassifier()),
    ])

BOW_bi_rf_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', RandomForestClassifier()),
    ])

# TF-IDF - bi-grams
TFIDF_bi_svm_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
    ])

TFIDF_bi_nb_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
    ])

TFIDF_bi_lr_pipe = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', LogisticRegression()),
    ])

TFIDF_bi_cnb_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 2))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2)),
  ('clf', ComplementNB()),
  ])

TFIDF_bi_dt_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 2))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2)),
  ('clf', DecisionTreeClassifier()),
  ])

TFIDF_bi_bg_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 2))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2)),
  ('clf', BaggingClassifier()),
  ])

TFIDF_bi_rf_pipe = Pipeline([
  ('vect', CountVectorizer(ngram_range=(1, 2))),
  ('tfidf', TfidfTransformer()),
  ('fselect', SelectKBest(chi2)),
  ('clf', RandomForestClassifier()),
  ])

# ------------------------- nested dictionary with all pipes and parameters for exhaustive search ------------------------- #
pipes_params = {    #BOW_uni_svm_pipe, BOW_bi_svm_pipe, TFIDF_uni_svm_pipe, TFIDF_bi_svm_pipe
    "svm": {"pipes": [TFIDF_uni_svm_pipe], 
             "grid_params": {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                             'clf__gamma': [1, 0.1, 0.01, 0.001],
                             #'vect__max_features': [int(VOCABULARY_SIZE*0.90), int(VOCABULARY_SIZE*0.80), None],
                             'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                             'fselect__k': [1000, 2000, 2500, 3000, 3500, 3700, "all"]}
    },
    "nb": {"pipes": [BOW_uni_nb_pipe, BOW_bi_nb_pipe, TFIDF_uni_nb_pipe, TFIDF_bi_nb_pipe], 
           "grid_params": {'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                           'fselect__k': [1000, 2000, 3000, 3500, 4000, "all"]}
    },
    "lr": {"pipes": [BOW_uni_lr_pipe, BOW_bi_lr_pipe, TFIDF_uni_lr_pipe, TFIDF_bi_lr_pipe], 
           "grid_params": {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                           'clf__max_iter': [1500],
                           'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                           'fselect__k': [1000, 2000, 3000, 3500, 4000, "all"]}
    },
    "cnb": {"pipes": [BOW_uni_cnb_pipe, BOW_bi_cnb_pipe, TFIDF_uni_cnb_pipe, TFIDF_bi_cnb_pipe],
           "grid_params": {'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                           'fselect__k': [1000, 2000, 3000, 3500, 4000, "all"]}
    },
    "dt": {"pipes": [BOW_uni_dt_pipe, BOW_bi_dt_pipe, TFIDF_uni_dt_pipe, TFIDF_bi_dt_pipe],
        "grid_params": {'clf__criterion': ['gini', 'entropy'],
                        #'clf__max_depth': [2,4,6,8,10,12],
                        #'clf__min_samples_split': range(2,10),
                        #'clf__min_samples_leaf': range(1,5),
                        'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                        'fselect__k': [1000, 2000, 3000, 3500, 4000, "all"]}
    },
    "bg": {"pipes": [TFIDF_uni_bg_dt_pipe, TFIDF_uni_bg_svm_pipe, TFIDF_uni_bg_lr_pipe],
        "grid_params": {'clf__n_estimators': [10, 30, 50, 100],
                        'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                        'fselect__k': [1000, 2000, 3000, 3500, 4000, "all"]}
    },
    "rf": {"pipes": [BOW_uni_rf_pipe],
        "grid_params": {'clf__criterion': ["gini", "entropy"],
                        'clf__n_estimators': [100, 300, 500, 750, 800, 1200],
                        'vect__max_df': (0.65, 0.75, 0.85, 1.0),
                        'clf__min_samples_split': [2, 5, 10],
                        'fselect__k': [1000, 2000, 3000, 3500, 4000, "all"]}
    }
}

#print(svm.SVC().get_params().keys())
#print(MultinomialNB().get_params().keys())
#print(ComplementNB().get_params().keys())
#print(LogisticRegression().get_params().keys())
#print(SelectKBest(chi2).get_params().keys())
#print(DecisionTreeClassifier().get_params().keys())

svm_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

nb_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

lr_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

cnb_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

dt_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

bg_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

rf_results = {"accuracy mean scores": [],
               "accuracy std scores": [],
               "precision mean scores": [],
               "precision std scores": [],
               "recall mean scores": [],
               "recall std scores": [],
               "f1_score mean scores": [],
               "f1_score std scores": [],
               "best params": []}

# ------------------------- grid searching best hyper-parameters for each classifier ------------------------- #
program_start = time.time()

i = 0
for elem in pipes_params:
    i += 1
    if elem == "svm":
        svm_search_start = time.time()
        print("Performing grid search on svm...")
        model_counter = 0
        for pipe in pipes_params["svm"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["svm"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            svm_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                svm_results["%s mean scores" % scorer].append(best_mean_score)
                svm_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/SVM_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        svm_search_stop = time.time()
        duration_run = round((svm_search_stop-svm_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
    """
    if elem == "nb":
        nb_search_start = time.time()
        print("Performing grid search on nb...")
        model_counter = 0
        for pipe in pipes_params["nb"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["nb"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            nb_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                nb_results["%s mean scores" % scorer].append(best_mean_score)
                nb_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/NB_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        nb_search_stop = time.time()
        duration_run = round((nb_search_stop-nb_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
    
    if elem == "lr":
        lr_search_start = time.time()
        print("Performing grid search on lr...")
        model_counter = 0
        for pipe in pipes_params["lr"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["lr"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            lr_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                lr_results["%s mean scores" % scorer].append(best_mean_score)
                lr_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/LR_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        lr_search_stop = time.time()
        duration_run = round((lr_search_stop-lr_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
    
    if elem == "cnb":
        cnb_search_start = time.time()
        print("Performing grid search on cnb...")
        model_counter = 0
        for pipe in pipes_params["cnb"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["cnb"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            cnb_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                cnb_results["%s mean scores" % scorer].append(best_mean_score)
                cnb_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/CNB_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        cnb_search_stop = time.time()
        duration_run = round((cnb_search_stop-cnb_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
    
    if elem == "dt":
        dt_search_start = time.time()
        print("Performing grid search on dt...")
        model_counter = 0
        for pipe in pipes_params["dt"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["dt"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            dt_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                dt_results["%s mean scores" % scorer].append(best_mean_score)
                dt_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/DT_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        dt_search_stop = time.time()
        duration_run = round((dt_search_stop-dt_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
    
    if elem == "bg":
        bg_search_start = time.time()
        print("Performing grid search on bg...")
        model_counter = 0
        for pipe in pipes_params["bg"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["bg"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            bg_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                bg_results["%s mean scores" % scorer].append(best_mean_score)
                bg_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/BG_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        bg_search_stop = time.time()
        duration_run = round((bg_search_stop-bg_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
    
    if elem == "rf":
        rf_search_start = time.time()
        print("Performing grid search on rf...")
        model_counter = 0
        for pipe in pipes_params["rf"]["pipes"]:
            print("pipeline:", [name for name, _ in pipe.steps])
            params = pipes_params["rf"]["grid_params"]
            X, y = shuffle(lemmatized_dataset, labels, random_state=i*320391)
            clf = GridSearchCV(pipe, params, scoring=scoring, cv=10, refit="accuracy", n_jobs=-1)
            clf.fit(X, y)
            rf_results["best params"].append(clf.best_params_)

            #getting metrics results 
            results = clf.cv_results_            
            for scorer in scoring:
                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_mean_score = results['mean_test_%s' % scorer][best_index]
                best_std_score = results['std_test_%s' % scorer][best_index]
                rf_results["%s mean scores" % scorer].append(best_mean_score)
                rf_results["%s std scores" % scorer].append(best_std_score)

            # saving best estimator
            model_name =  './Tuned_Models/RF_' + str(model_counter) + ".pkl"
            joblib.dump(clf, model_name, compress = 1)
            model_counter += 1
        rf_search_stop = time.time()
        duration_run = round((rf_search_stop-rf_search_start)/60, 2)
        print('done in {} mins'.format(duration_run))
        """
program_stop = time.time()
duration_total = round((program_stop-program_start)/60, 2)
print('Total duration in {} mins'.format(duration_total))

# ------------------------- print results ------------------------- #
print(svm_results)
#print(nb_results)
#print(lr_results)
#print(cnb_results)
#print(dt_results)
#print(bg_results)
#print(rf_results)
"""
# ------------------------- save results on files ------------------------- #
with open('./MTT_Results/MTT_svm_results', 'w') as fout:
    json.dump(svm_results, fout, indent=4)

with open('./MTT_Results/MTT_nb_results', 'w') as fout:
    json.dump(nb_results, fout, indent=4)

with open('./MTT_Results/MTT_lr_results', 'w') as fout:
    json.dump(lr_results, fout, indent=4)

with open('./MTT_Results/MTT_cnb_results', 'w') as fout:
    json.dump(cnb_results, fout, indent=4)

with open('./MTT_Results/MTT_dt_results', 'w') as fout:
    json.dump(dt_results, fout, indent=4)

with open('./MTT_Results/MTT_bg_results', 'w') as fout:
    json.dump(bg_results, fout, indent=4)

with open('./MTT_Results/MTT_rf_results', 'w') as fout:
    json.dump(rf_results, fout, indent=4)
"""