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
    select type, count(DISTINCT content)
	    from article
        inner join has_type ht on article.id = ht.article_id
        inner join type t on ht.type_id = t.id
    group by type
""")
except:
    # rollback in the case of stuck transaction
    print("rolled back")
    conn.rollback()

dict_Entire_Fakenews = {'fake': 928083/9408908, 
                        'satire': 146080/9408908,
                        'bias': 1300444/9408908,
                        'conspiracy': 905981/9408908,
                        'junksci': 144939/9408908,
                        'hate': 117374/9408908,
                        'clickbait': 292201/9408908,
                        'unreliable': 319830/9408908,
                        'reliable': 1920139/9408908,
                        'political': 2435471/9408908,
                        }
df_fakeNewsCorpus = pd.DataFrame(dict_Entire_Fakenews.items(), columns=['Types', 'Entire FakeNewsCorpus'])


df = pd.DataFrame(cur.fetchall(), columns=['Types', 'Count'])
Total = df['Count'].sum()
df['Our 100000 sample'] = df['Count']/Total
df= df.join(df_fakeNewsCorpus.set_index('Types'), on='Types')

# plotting a bar graph
labels = ['Our sub sample', 'Entire FakeNewsCorpus']

#plt.hist(df['Types'],df[['Proportion', 'Proportion FakeNewsCorpus']], 12, density=True, histtype='bar', color=['blue', 'yellow'], label=labels)
#plt.legend(prop={'size': 10})
#plt.set_title('bars with legend')
#plt.show()


df.plot(x="Types", y=["Our 100000 sample", "Entire FakeNewsCorpus"], kind="bar", xlabel = 'Type', ylabel = 'Frequency', width=0.6)
plt.show()


# Close communication with the database
cur.close()
conn.close()
