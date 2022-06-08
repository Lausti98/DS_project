drop table if exists wiki_article;

CREATE TABLE wiki_article(
    id varchar(50) NOT NULL,
    art_date varchar(50) null,
    content TEXT NOT NULL, 
    art_URL varchar(200) NOT NULL,
    PRIMARY KEY (id)
);
UPDATE wiki_article SET art_date=TO_DATE(art_date,'YYYY-MM-DD');