import numpy as np
import pandas as pd
from pprint import pprint
from time import time
import logging
import psycopg2 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# Connect with database
conn = psycopg2.connect("host=localhost dbname=sample_db user=anderssteiness password=XXX")
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
        where t.type = 'reliable' or t.type = 'political' or t.type = 'fake' or t.type = 'bias'
""")
except:
    #rollback in the case of stuck transaction
    print("rolled back")
    conn.rollback() 

df = pd.DataFrame(cur.fetchall(), columns = ['id', 'title', 'type', 'domain', 
                                             'article_url', 'scraped_at', 'inserted_at',
                                             'updated_at', 'authors', 'keywords', 'tags',
                                             'content'])

#if blank values are NaN first replace to ''
df = df.fillna('')

#we group by id, since from our query, each article with e.g. tags will appear for each
#tag seperately, we implode on all columns using groupby function on all the attributes that
#will be the same for each duplicate and we use aggregate function for the features/columns
#that causes the duplication of tuples which include keywords and tags. Notice that keywords
#and tags are being duplicated themselves due to cross-product, thus we create a function
#that only includes unique values and does not contain '' value.
df = df.groupby(['id', 'title', 'type','domain', 'article_url', 'scraped_at', 'inserted_at', 
                                             'updated_at', 'content']).agg({'keywords': lambda x: [x for x in list(set(x.tolist())) if str(x) != ''], 'tags': lambda x: [x for x in list(set(x.tolist())) if str(x) != '']}  
                                             ).reset_index()

df.loc[df['type'] == 'political', 'type'] = 'reliable'
df.loc[df['type'] == 'bias', 'type'] = 'reliable'
print(df)


#data splitting into training and test - only 1 feature which is 'content'
folds = 4
x = df['content']
y = df['type']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.8, random_state=0, stratify=y)

#tfidf_pipeline, transformer pipeline. Vectorization of # of words and normalization.
# tfidf_pipeline = Pipeline([
#     ("vect", CountVectorizer()),
#     ('tfidf', TfidfTransformer())
# ])





########### support vector machine classifier ###########
svm_pipeline = Pipeline([
    #('tfidf_pipeline', tfidf_pipeline),
    ("vect", CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('svc', LinearSVC())
])

parameters = {
    "vect__max_df": (0.5, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'svc__C': (1, 10)
}


# Find the best parameters for both the feature extraction and the
# classifier

grid_search = GridSearchCV(svm_pipeline, parameters, verbose=1) # , n_jobs=-1

print("Performing grid search...")
print("pipeline:", [name for name, _ in svm_pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(X_train[:5000], Y_train[:5000]) # we only take a sample due to computation time
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
 

#fitting and prediction LinearSVC model with best parameters on test set

LinearSVC_classifier = Pipeline([
    ("vect", CountVectorizer(ngram_range = (1, 2), max_df = 1.0)),
    ('tfidf', TfidfTransformer(norm = 'l2')), #cosine similarity
    ('svc', LinearSVC(C = 10))
])
 
LinearSVC_classifier.fit(X_train, Y_train)
LinearSVC_y_predicted = LinearSVC_classifier.predict(X_test)
from sklearn.metrics import f1_score
print("LinearSVC f1_score: ", str(f1_score(Y_test, LinearSVC_y_predicted, average ='weighted'))) #OBS! Look at weighted






########### K-nearest neighbors classifier ###########

KNN_pipeline = Pipeline([
    #('tfidf_pipeline', tfidf_pipeline),
    ("vect", CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('KNN', KNeighborsClassifier())
])

KNN_parameters = {
    "vect__max_df": (0.5, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'KNN__n_neighbors': (1, 3, 5)
}


# Find the best parameters for both the feature extraction and the
# classifier
KNN_grid_search = GridSearchCV(KNN_pipeline, KNN_parameters, verbose=1) # , n_jobs=-1

print("Performing grid search...")
print("pipeline:", [name for name, _ in KNN_pipeline.steps])
print("parameters:")
pprint(KNN_parameters)
t0 = time()
KNN_grid_search.fit(X_train[:5000], Y_train[:5000]) # we only take a sample due to computation time
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % KNN_grid_search.best_score_)
print("Best parameters set:")
KNN_best_parameters = KNN_grid_search.best_estimator_.get_params()
for param_name in sorted(KNN_parameters.keys()):
    print("\t%s: %r" % (param_name, KNN_best_parameters[param_name]))


#fitting and prediction LinearSVC model with best parameters on test set
KNN_classifier = Pipeline([
    ("vect", CountVectorizer(ngram_range = (1, 2), max_df = 1.0)),
    ('tfidf', TfidfTransformer(norm = 'l2')), #cosine similarity
    ('svc', KNeighborsClassifier(n_neighbors = 5))
])
 
KNN_classifier.fit(X_train, Y_train)
KNNC_y_predicted = KNN_classifier.predict(X_test)
from sklearn.metrics import f1_score
print("KNN f1_score: ", str(f1_score(Y_test, KNNC_y_predicted, average ='weighted'))) #OBS! Look at weighted

















# Make the changes to the database persistent
conn.commit()

# Close communication with the database
cur.close()
conn.close()





