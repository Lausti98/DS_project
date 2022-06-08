
echo "Write database name"
read DATABASE 
echo "Write username"
read USERNAME
echo "Write SQL file"
read SQL

psql -d $DATABASE -U $USERNAME -f $SQL -W

python3 populatedb_wiki.py