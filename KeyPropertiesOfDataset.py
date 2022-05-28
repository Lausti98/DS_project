import matplotlib.pyplot as plt
import pandas as pd
import psycopg2

# Connect with database
conn = psycopg2.connect(
    "host=localhost dbname=sample_db user=anderssteiness password=XXX")
cur = conn.cursor()

# A query of all articles with all entities joined
try:
    cur.execute("""
    select type, count(*)
	    from article
        inner join has_type ht on article.id = ht.article_id
        inner join type t on ht.type_id = t.id
    group by type
""")
except:
    # rollback in the case of stuck transaction
    print("rolled back")
    conn.rollback()

df = pd.DataFrame(cur.fetchall(), columns=['Types', 'Count'])

# plotting a bar graph
df.plot(x="Types", y="Count", kind="bar")
plt.show()


# Close communication with the database
cur.close()
conn.close()
