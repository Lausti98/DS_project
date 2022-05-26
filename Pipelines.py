from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

Pipelines = {
    'svm_pipeline': Pipeline([
        ("vect", CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svc', LinearSVC())
    ]),
    'svm_parameters': {
        "vect__max_df": (0.5, 1.0),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'),
        'svc__C': (1, 10)
    }
}
