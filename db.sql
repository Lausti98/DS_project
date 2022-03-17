CREATE TABLE Article(
    Article_ID int NOT NULL, 
    Domain varchar(50), 
    Article_Url varchar(200) NOT NULL,
    Content varchar(100000) NOT NULL,
    Title varchar(200) NOT NULL, 
    Meta_description_object varchar(200), 
    Scraped_At DATE NOT NULL, 
    Inserted_At DATE NOT NULL,   
    Updated_At DATE NOT NULL, 
    PRIMARY KEY (Article_ID));

CREATE TABLE Type(
    Type_ID int NOT NULL, 
    Type varchar(50), 
    PRIMARY KEY (Type_ID));

CREATE TABLE Has_type(
    Type_ID int NOT NULL,
    Article_ID int NOT NULL, 
    PRIMARY KEY (Type_ID, Article_ID),
    FOREIGN KEY (Type_ID) REFERENCES Type,
    FOREIGN KEY (Article_ID) REFERENCES Article
);