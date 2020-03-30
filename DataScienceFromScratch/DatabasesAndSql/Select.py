import CreateTableAndInsert as cr

# SELECT * FROM users;
print(cr.users.select())

# SELECT * FROM users LIMIT 2;
print(cr.users.limit(2))

# SELECT user_id FROM users;
print(cr.users.select(keep_columns=["user_id"]))

# SELECT user_id FROM users WHERE name = 'Dunn';
print(cr.users.where(lambda row: row["name"] == "Dunn") \
             .select(keep_columns=["user_id"]))

# SELECT LENGTH(name) AS name_length FROM users;
def name_length(row): return len(row["name"])

print(cr.users.select(keep_columns=[],
            additional_columns = { "name_length" : name_length }))
