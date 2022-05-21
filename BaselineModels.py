import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import psycopg2

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
        where t.type = 'reliable' or t.type = 'political' or t.type = 'fake' or t.type = 'bias'
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
                 'updated_at', 'content']).agg({'keywords': lambda x: [x for x in list(set(x.tolist())) if str(x) != ''], 'tags': lambda x: [x for x in list(set(x.tolist())) if str(x) != '']}
                                               ).reset_index()

print(df)


# tfidf_pipeline, transformer pipeline. Vectorization of # of words and normalization.
tfidf_pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ('tfidf', TfidfVectorizer())
])


# Make the changes to the database persistent
conn.commit()

# Close communication with the database
cur.close()
conn.close()
