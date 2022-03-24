
echo "Write database name"
read DATABASE 
echo "Write username"
read USERNAME

psql -d $DATABASE -U $USERNAME -f Generated_SQL_Schema_DS.sql -W

python3 populatedb.py