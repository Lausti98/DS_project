from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
from pprint import pprint
from time import time
import logging
import psycopg2
from Pipelines import Pipelines

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


# Connect with database
conn = psycopg2.connect(
    "host=localhost dbname=sample_db user=anderssteiness password=XXX")
cur = conn.cursor()

# A query of all articles with all entities joined
try:
    cur.execute("""
    select article.id, title, type, domain, article_url, scraped_at, inserted_at, updated_at, a.name, keyword, tag, content  
	    from article
        inner join has_type ht on article.id = ht.article_id
        inner join type t on ht.type_id = t.id
	    left join written_by wb on article.id = wb.article_id
        left join author a on wb.author_id = a.id
	    left join has_tag hta on article.id = hta.article_id
        left join tag ta on hta.tag_id = ta.id
	    left join has_meta_keyword hmk on article.id = hmk.article_id
        left join meta_keyword mk on hmk.meta_keyword_id = mk.id
        where t.type = 'reliable' or t.type = 'political' or t.type = 'fake' or t.type = 'clickbait'
""")
except:
    # rollback in the case of stuck transaction
    print("rolled back")
    conn.rollback()

df = pd.DataFrame(cur.fetchall(), columns=['id', 'title', 'type', 'domain',
                                           'article_url', 'scraped_at', 'inserted_at',
                                           'updated_at', 'authors', 'keywords', 'tags',
                                           'content'])

# if blank values are NaN first replace to ''
df = df.fillna('')

# we group by id, since from our query, each article with e.g. tags will appear for each
# tag seperately, we implode on all columns using groupby function on all the attributes that
# will be the same for each duplicate and we use aggregate function for the features/columns
# that causes the duplication of tuples which include keywords and tags. Notice that keywords
# and tags are being duplicated themselves due to cross-product, thus we create a function
# that only includes unique values and does not contain '' value.
df = df.groupby(['id', 'title', 'type', 'domain', 'article_url', 'scraped_at', 'inserted_at',
                 'updated_at', 'content']).agg({'keywords': lambda x: [x for x in list(set(x.tolist())) if str(x) != ''],
                                                'tags': lambda x: [x for x in list(set(x.tolist())) if str(x) != ''],
                                                'authors': lambda x: [x for x in list(set(x.tolist())) if str(x) != '']}
                                               ).reset_index()

#df.loc[df['type'] == 'political', 'type'] = 'real'
#df.loc[df['type'] == 'clickbait', 'type'] = 'real'
#df.loc[df['type'] == 'reliable', 'type'] = 'real'
df['Fake or Real'] = np.where(df['type'] == 'fake', 1, 0)  # fake = 1, real = 0


# data splitting into training and test - only 1 feature which is 'content'
folds = 4
x = df['content']
y = df['type']
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.8, random_state=0, stratify=y)


svm_pipe = Pipelines['svm_pipeline']
svm_params = Pipelines['svm_parameters']

# data splitting into training and test - only 1 feature which is 'content'
# x = df['content'] #single feature
#y = df['Fake or Real']

# creating train and test set for mutiple feature simple models
df['Multiple Features'] = df['title'] + df['content'] + df['domain'] + df['authors'].apply(lambda x: ','.join(map(str, x))).str.lower().str.replace(" ", "-")

x = df['Multiple Features'][:10000]
y = df['Fake or Real'][:10000]

print(x)

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y)


# ########### support vector machine classifier ###########
# svm_pipeline = Pipeline([
#     #('tfidf_pipeline', tfidf_pipeline),
#     ("vect", CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('svc', LinearSVC())
# ])

# parameters = {
#     "vect__max_df": (0.5, 1.0),
#     # 'vect__max_features': (None, 5000, 10000, 50000),
#     "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
#     # 'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'svc__C': (1, 10)
# }


# # Find the best parameters for both the feature extraction and the
# # classifier

svm_grid_search = GridSearchCV(svm_pipe, svm_params, verbose=1)  # , n_jobs=-1

print("Performing grid search...")
print("pipeline:", [name for name, _ in svm_pipe.steps])
print("parameters:")
pprint(svm_params)
t0 = time()
# we only take a sample due to computation time
svm_grid_search.fit(X_train[:5000], Y_train[:5000])
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % svm_grid_search.best_score_)
print("Best parameters set:")
best_parameters = svm_grid_search.best_estimator_.get_params()
for param_name in sorted(svm_params.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# fitting and prediction LinearSVC model with best parameters on test set
LinearSVC_y_predicted = svm_grid_search.predict(X_test)
# , average ='weighted'
print("LinearSVC f1_score: ", str(f1_score(Y_test, LinearSVC_y_predicted)))
# LinearSVC f1_score:  0.9780560819532559


# multiple feature fit
LinearSVC_y_predicted = LinearSVC.predict(X_test)
print("LinearSVC f1_score multiple features: ",
      str(f1_score(Y_test, LinearSVC_y_predicted)))


########### K-nearest neighbors classifier ###########

KNN_pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('KNN', KNeighborsClassifier())
])

KNN_parameters = {
    "vect__max_df": (0.5, 1.0),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__norm': ('l1', 'l2'),
    'KNN__n_neighbors': (1, 3, 5)
}

# Find the best parameters for both the feature extraction and the
# classifier
KNN_grid_search = GridSearchCV(
    KNN_pipeline, KNN_parameters, verbose=1)  # , n_jobs=-1

print("Performing grid search...")
print("pipeline:", [name for name, _ in KNN_pipeline.steps])
print("parameters:")
pprint(KNN_parameters)
t0 = time()
# we only take a sample due to computation time
KNN_grid_search.fit(X_train[:10000], Y_train[:10000])
#print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % KNN_grid_search.best_score_)
print("Best parameters set:")
KNN_best_parameters = KNN_grid_search.best_estimator_.get_params()
for param_name in sorted(KNN_parameters.keys()):
    print("\t%s: %r" % (param_name, KNN_best_parameters[param_name]))


# fitting and prediction KNN model with best parameters on test set
KNN_y_predicted = KNN_grid_search.predict(X_test)
print("KNN f1_score: ", str(f1_score(Y_test, KNN_y_predicted)))
# KNN f1_score:  0.865467192829911

# multiple feature fit
#KNN_y_predicted = KNeighborsClassifier.predict(X_test)
#print("KNN f1_score multiple features: ", str(f1_score(Y_test, KNN_y_predicted)))


######## Random forest classifier ############

RF_pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('RF', RandomForestClassifier())
])

RF_parameters = {
    "vect__max_df": (0.5, 1.0),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__norm': ('l1', 'l2'),
    'RF__n_estimators': (100, 300),
    'RF__max_depth': (None, 30)
}


# Find the best parameters for both the feature extraction and the
# classifier
RF_grid_search = GridSearchCV(
    RF_pipeline, RF_parameters, verbose=1)  # , n_jobs=-1

print("Performing grid search...")
print("pipeline:", [name for name, _ in RF_pipeline.steps])
print("parameters:")
pprint(RF_parameters)
t0 = time()
# we only take a sample due to computation time
RF_grid_search.fit(X_train[:10000], Y_train[:10000])
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % RF_grid_search.best_score_)
print("Best parameters set:")
RF_best_parameters = RF_grid_search.best_estimator_.get_params()
for param_name in sorted(RF_parameters.keys()):
    print("\t%s: %r" % (param_name, RF_best_parameters[param_name]))


# fitting and prediction Random Forest model with best parameters on test set
RF_y_predicted = RF_grid_search.predict(X_test)
print("RF f1_score: ", str(f1_score(Y_test, RF_y_predicted)))
# RF f1_score:  0.9711761278170016, but between 0.94-0.97

# multiple feature fit
#RF_y_predicted = RandomForestClassifier.predict(X_test)
#print("Random Forest f1_score multiple features: ", str(f1_score(Y_test, RF_y_predicted)))


# # fitting and prediction LinearSVC model with best parameters on test set

# LinearSVC_classifier = Pipeline([
#     ("vect", CountVectorizer(ngram_range=(1, 2), max_df=1.0)),
#     ('tfidf', TfidfTransformer(norm='l2')),  # cosine similarity
#     ('svc', LinearSVC(C=10))
# ])

# LinearSVC_classifier.fit(X_train, Y_train)
# LinearSVC_y_predicted = LinearSVC_classifier.predict(X_test)
# print("LinearSVC f1_score: ", str(f1_score(
#     Y_test, LinearSVC_y_predicted, average='weighted')))  # OBS! Look at weighted
######## Logistic Regression classifier ############

LR_pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('LR', LogisticRegression(solver='lbfgs', max_iter=500))
])

LR_parameters = {
    "vect__max_df": (0.5, 1.0),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__norm': ('l1', 'l2'),
    'LR__C': (1, 10, 20)
}

# Find the best parameters for both the feature extraction and the
# classifier
LR_grid_search = GridSearchCV(
    LR_pipeline, LR_parameters, verbose=1)  # , n_jobs=-1

print("Performing grid search...")
print("pipeline:", [name for name, _ in LR_pipeline.steps])
print("parameters:")
pprint(LR_parameters)
t0 = time()
# we only take a sample due to computation time
LR_grid_search.fit(X_train[:5000], Y_train[:5000])
#print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % LR_grid_search.best_score_)
print("Best parameters set:")
LR_best_parameters = LR_grid_search.best_estimator_.get_params()
for param_name in sorted(LR_parameters.keys()):
    print("\t%s: %r" % (param_name, LR_best_parameters[param_name]))

# fitting and prediction Logistic Regression model with best parameters on test set
LR_y_predicted = LR_grid_search.predict(X_test)
print("LR f1_score: ", str(f1_score(Y_test, LR_y_predicted)))
# LR f1_score:  0.9746363910268702

# ########### K-nearest neighbors classifier ###########

# KNN_pipeline = Pipeline([
#     #('tfidf_pipeline', tfidf_pipeline),
#     ("vect", CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('KNN', KNeighborsClassifier())
# ])
# multiple feature fit
#LR_y_predicted = LogisticRegression.predict(X_test)
#print("Logistic Regression f1_score multiple features: ", str(f1_score(Y_test, LR_y_predicted)))

# KNN_parameters = {
#     "vect__max_df": (0.5, 1.0),
#     # 'vect__max_features': (None, 5000, 10000, 50000),
#     "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
#     # 'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'KNN__n_neighbors': (1, 3, 5)
# }


# # Find the best parameters for both the feature extraction and the
# # classifier
# KNN_grid_search = GridSearchCV(
#     KNN_pipeline, KNN_parameters, verbose=1)  # , n_jobs=-1

# print("Performing grid search...")
# print("pipeline:", [name for name, _ in KNN_pipeline.steps])
# print("parameters:")
# pprint(KNN_parameters)
# t0 = time()
# # we only take a sample due to computation time
# KNN_grid_search.fit(X_train[:5000], Y_train[:5000])
# print("done in %0.3fs" % (time() - t0))
# print()

# print("Best score: %0.3f" % KNN_grid_search.best_score_)
# print("Best parameters set:")
# KNN_best_parameters = KNN_grid_search.best_estimator_.get_params()
# for param_name in sorted(KNN_parameters.keys()):
#     print("\t%s: %r" % (param_name, KNN_best_parameters[param_name]))


# # fitting and prediction LinearSVC model with best parameters on test set
# KNN_classifier = Pipeline([
#     ("vect", CountVectorizer(ngram_range=(1, 2), max_df=1.0)),
#     ('tfidf', TfidfTransformer(norm='l2')),  # cosine similarity
#     ('svc', KNeighborsClassifier(n_neighbors=5))
# ])

# KNN_classifier.fit(X_train, Y_train)
# KNNC_y_predicted = KNN_classifier.predict(X_test)
# print("KNN f1_score: ", str(f1_score(Y_test, KNNC_y_predicted,
#       average='weighted')))  # OBS! Look at weighted


# ########## SGD CLASSIFIER #####################################
# # Creating SGD pipeline
# SGD_pipeline = Pipeline([
#     ("vect", CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('SGD', SGDClassifier())
# ])

# SGD_parameters = {
#     "vect__max_df": (0.5, 1.0),
#     # 'vect__max_features': (None, 5000, 10000, 50000),
#     "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
#     # 'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'SGD__max_iter': (20, 30),
#     'SGD__alpha': (0.01, 0.001),
#     'SGD__penalty': ('l1', 'l2')
# }

# # Find the best parameters for both the feature extraction and the
# # classifier
# SGD_grid_search = GridSearchCV(
#     SGD_pipeline, SGD_parameters, verbose=1)  # , n_jobs=-1

# print("Performing grid search...")
# print("pipeline:", [name for name, _ in SGD_pipeline.steps])
# print("parameters:")
# pprint(SGD_parameters)
# t0 = time()
# # we only take a sample due to computation time
# SGD_grid_search.fit(X_train[:5000], Y_train[:5000])
# print("done in %0.3fs" % (time() - t0))
# print()

# print("Best score: %0.3f" % SGD_grid_search.best_score_)
# print("Best parameters set:")
# SGD_best_parameters = SGD_grid_search.best_estimator_.get_params()
# for param_name in sorted(SGD_parameters.keys()):
#     print("\t%s: %r" % (param_name, SGD_best_parameters[param_name]))


# Close communication with the database
cur.close()
conn.close()
