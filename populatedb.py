import psycopg2 
import pandas as pd 

conn = psycopg2.connect("host=localhost dbname=moviedb user=laust1 password=123")
cur = conn.cursor()

def copy_f_to_table(filepath, table, cur=cur):
    with open (filepath, 'r') as f:
        next(f)
        cur.copy_from(f, table, sep='\t')

d_path = 'Data_Files/'

copy_f_to_table(d_path + 'Article.csv', 'article')

copy_f_to_table(d_path + 'Authors_clean.csv', 'author')

copy_f_to_table(d_path + 'Tag_clean.csv', 'tag')

copy_f_to_table(d_path + 'Meta_keyword_clean.csv', 'meta_keyword')

copy_f_to_table(d_path + 'Type_clean.csv', 'type')

copy_f_to_table(d_path + 'WrittenBy.csv', 'written_by')

copy_f_to_table(d_path + 'Has_meta_keyword.csv', 'has_meta_keyword')

copy_f_to_table(d_path + 'Has_type.csv', 'has_type')

copy_f_to_table(d_path + 'Has_tag.csv', 'has_tag')






# with open ('Data_Files/Article.csv', 'r') as f:
#     next(f)
#     cur.copy_from(f, 'article', sep='\t')

# with open ('Data_Files/Authors_clean.csv', 'r') as f:
#     next(f)
#     cur.copy_from(f, 'author', sep='\t')

# with open ('Data_Files/Tag_clean.csv', 'r') as f:
#     next(f)
#     cur.copy_from(f, 'tag', sep='\t')

# with open ('Data_Files/Meta_keyword_clean.csv', 'r') as f:
#     next(f)
#     cur.copy_from(f, 'meta_keyword', sep='\t')

# with open ('Data_Files/Type_clean.csv', 'r') as f:
#     next(f)
#     cur.copy_from(f, 'type', sep='\t')


# Make the changes to the database persistent
conn.commit()

# Close communication with the database
cur.close()
conn.close()
