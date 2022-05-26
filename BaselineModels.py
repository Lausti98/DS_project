from matplotlib.pyplot import grid
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
def run_model(pipeline, parameters, model_name):
    grid_search = GridSearchCV(pipeline, parameters, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    # we only take a sample due to computation time
    grid_search.fit(X_train[:5000], Y_train[:5000])
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # predict y and compute f1 score
    predictions = grid_search.predict(X_test)
    print(f'f1 score of {model_name}: {str(f1_score(Y_test, predictions))}')


# creating train and test set for mutiple feature simple models
# y = df['Fake or Real'][:10000]
# X_train, X_test, Y_train, Y_test = train_test_split(
#     x, y, test_size=0.2, random_state=0, stratify=y)


########### SVC CLASSIFIER ###############
svc_pipe = Pipelines['svc_pipeline']
svc_params = Pipelines['svc_parameters']
run_model(svc_pipe, svc_params, 'SVC')


########### K-nearest neighbors classifier ###########
knn_pipe = Pipelines['KNN_pipeline']
knn_params = Pipelines['KNN_parameters']
run_model(knn_pipe, knn_params, 'KNN')


######## Random forest classifier ############
rf_pipe = Pipelines['RF_pipeline']
rf_params = Pipelines['RF_parameters']
run_model(rf_pipe, rf_params, 'Random Forest')


######## Logistic Regression classifier ############
lr_pipe = Pipelines['LR_pipeline']
lr_params = Pipelines['LR_parameters']
run_model(lr_pipe, lr_params, 'Logistic Regression')


########## SGD CLASSIFIER #####################################
sgd_pipe = Pipelines['SGD_pipeline']
sgd_params = Pipelines['SGD_parameters']
run_model(sgd_pipe, sgd_params, 'SGD')


# Close communication with the database
cur.close()
conn.close()
