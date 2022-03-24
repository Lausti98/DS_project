# DS_project

To add database script to local database type command in terminal: 
    psql -d <dbname> -U <user> -f Generated_SQL_Schema_DS.sql -W

To populate database: 
    python populatedb.py

Push to git: 
    1. git pull 
    2. git add .
    3. git commit -m"commit message"
    4. git push
