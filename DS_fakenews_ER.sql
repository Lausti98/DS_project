drop table if exists Article;
drop table if exists Tag;
drop table if exists Type;
drop table if exists Author;
drop table if exists Meta_keyword;
drop table if exists Keyword; 

 

CREATE TABLE Article(
    Article_ID int NOT NULL, 
    Domain varchar(50), 
    Article_Url varchar(200) NOT NULL,
    Content TEXT NOT NULL,
    Title varchar(200) NOT NULL, 
    Meta_description_object varchar(200), 
    Scraped_At DATE NOT NULL, 
    Inserted_At DATE NOT NULL,   
    Updated_At DATE NOT NULL, 
    PRIMARY KEY (Article_ID)
);


CREATE TABLE Tag(
    Tag_ID int NOT NULL,
    Tag varchar(50) NOT NULL, 
    PRIMARY KEY (Tag_ID)
);


CREATE TABLE Type(
    Type_ID int NOT NULL, 
    Type varchar(50), 
    PRIMARY KEY (Type_ID)
);

CREATE TABLE Author(
    Author_ID int NOT NULL, 
    Name varchar(50), 
    PRIMARY KEY (Author_ID)
);


CREATE TABLE Meta_keyword(
    Meta_KW_ID int NOT NULL, 
    Keyword varchar(50), 
    PRIMARY KEY (Meta_KW_ID)
);

CREATE TABLE Keyword(
    KW_ID int NOT NULL, 
    Keyword varchar(50), 
    PRIMARY KEY (KW_ID)
);


-- CREATE TABLE Has_type(
--     Type_ID int NOT NULL,
--     Article_ID int NOT NULL, 
--     PRIMARY KEY (Type_ID, Article_ID),
--     FOREIGN KEY (Type_ID) REFERENCES Type,
--     FOREIGN KEY (Article_ID) REFERENCES Article
-- );

