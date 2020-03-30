import CreateTableAndInsert as cr

cr.users.delete(lambda row: row["user_id"] == 1) # deletes rows with user_id == 1
print(cr.users)
cr.users.delete() # deletes every row
print(cr.users)
