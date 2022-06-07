from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

Pipelines = {
    'svc_pipeline': Pipeline([
        ("vect", CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svc', LinearSVC())
    ]),
    'svc_parameters': {
        "vect__max_df": (0.8),
        "vect__min_df": (10),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'),
        'svc__C': (1, 10, 30)
    },
    'KNN_pipeline': Pipeline([
        ("vect", CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('KNN', KNeighborsClassifier())
    ]),
    'KNN_parameters': {
        "vect__max_df": (0.8),
        "vect__min_df": (10),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'),
        'KNN__n_neighbors': (1, 3, 5)
    },
    'RF_pipeline': Pipeline([
        ("vect", CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('RF', RandomForestClassifier())
    ]),
    'RF_parameters': {
        "vect__max_df": (0.8),
        "vect__min_df": (10),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'),
        'RF__n_estimators': (100, 300),
        'RF__max_depth': (None, 30)
    },
    'LR_pipeline': Pipeline([
        ("vect", CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('LR', LogisticRegression(solver='lbfgs', max_iter=500))
    ]),
    'LR_parameters': {
        "vect__max_df": (0.8),
        "vect__min_df": (10),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'),
        'LR__C': (1, 10, 20)
    },
    'SGD_pipeline': Pipeline([
        ("vect", CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('SGD', SGDClassifier())
    ]),
    'SGD_parameters': {
        "vect__max_df": (0.8),
        "vect__min_df": (10),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'),
        'SGD__penalty': ('l1', 'l2'),
        # 'SGD__max_iter': (30)
    }



}
